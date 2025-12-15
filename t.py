#!/usr/bin/env python3
"""
strong-lens-mi300-train_infer.py
Final, high-speed script with pre-resizing cache and clean logs.
"""
import os, time, argparse, atexit, pickle, random, warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
import torchvision
from torchvision.transforms import v2 as T
from astropy.io import fits
from astropy.io.fits import BinTableHDU, ImageHDU, PrimaryHDU

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False

# --- Constants & Utilities ---
BANDS = ("g", "r", "i", "z", "y"); IMG_DTYPE = np.float32
def is_global_zero_env() -> bool: return int(os.environ.get("RANK", "0")) == 0
def rank_zero_print(msg: str):
    if is_global_zero_env(): print(msg, flush=True)
def _torch_world_size_safe() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized(): return dist.get_world_size()
    except Exception: pass
    return int(os.environ.get("WORLD_SIZE", "1"))
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# --- FITS I/O & Indexing ---
def _clean_numpy(x: np.ndarray) -> np.ndarray: return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
def load_5_band_image_from_parts(class_path: Path, base_name: str) -> Optional[np.ndarray]:
    image_bands = []
    try:
        for band in BANDS:
            file_path = class_path / f"{base_name}_{band}.fits"
            if not file_path.exists(): return None
            with fits.open(file_path, memmap=False) as hdul:
                found = False
                for hdu in hdul:
                    if isinstance(hdu, BinTableHDU) and f"band_{band}" in hdu.columns.names:
                        band_arr = np.asarray(hdu.data[f"band_{band}"][0], dtype=IMG_DTYPE)
                        image_bands.append(_clean_numpy(band_arr)); found = True; break
                    elif isinstance(hdu, (ImageHDU, PrimaryHDU)) and hasattr(hdu.data, 'ndim') and hdu.data.ndim >= 2:
                        band_arr = np.asarray(hdu.data, dtype=IMG_DTYPE)
                        image_bands.append(_clean_numpy(band_arr)); found = True; break
                if not found: return None
        if len(image_bands) != len(BANDS): return None
        img = np.stack(image_bands, axis=0); img = _clean_numpy(img)
        mean, std = img.mean(), img.std()
        if std > 1e-9: img = (img - mean) / std
        return img
    except Exception: return None
@dataclass
class Item: obj_id: str; class_path: str; base_name: str; label: int
def build_index(roots: List[str], label: int) -> List[Item]:
    all_items = []
    for root_str in roots:
        root = Path(root_str); rank_zero_print(f"[index] Scanning {root} for label={label}...")
        base_objects = {("_".join(p.stem.split("_")[:-1]), p.parent) for p in root.rglob("*.fits") if p.stem.split("_")[-1] in BANDS}
        for base, parent in base_objects:
            all_items.append(Item(obj_id=base, class_path=str(parent), base_name=base, label=label))
        rank_zero_print(f"[index] Found {len(base_objects):,} unique objects in {root}")
    return all_items

# --- Data Handling ---
def stratified_split(data_items: List[Item], val_split: float, seed: int) -> Tuple[List[Item], List[Item]]:
    labels = np.array([it.label for it in data_items]); pos_idx, neg_idx = np.where(labels == 1)[0], np.where(labels == 0)[0]
    rng = np.random.default_rng(seed); rng.shuffle(pos_idx); rng.shuffle(neg_idx)
    n_val_pos, n_val_neg = int(round(len(pos_idx) * val_split)), int(round(len(neg_idx) * val_split))
    val_idx = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
    train_idx = np.concatenate([pos_idx[n_val_pos:], neg_idx[n_val_neg:]])
    rng.shuffle(train_idx); rng.shuffle(val_idx)
    train = [data_items[i] for i in train_idx]; val = [data_items[i] for i in val_idx]
    rank_zero_print(f"[split] train={len(train):,} val={len(val):,} (stratified)")
    return train, val

class LensDataset(Dataset):
    def __init__(self, data_items: List[Item], augment: bool = False, image_size: int = 41):
        self.data_items, self.augment_flag = data_items, augment
        self.cache = {}; self.labels = torch.tensor([it.label for it in data_items], dtype=torch.float32)
        self.resize = T.Resize([image_size, image_size], antialias=True) if image_size != 41 else nn.Identity()

    def __len__(self): return len(self.data_items)
    def _maybe_augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.augment_flag: return x
        y = x.clone()
        if torch.rand(1) < 0.5: y = torch.flip(y, dims=(2,))
        if torch.rand(1) < 0.5: y = torch.flip(y, dims=(1,))
        k = torch.randint(0, 4, (1,)).item()
        if k > 0: y = torch.rot90(y, k=k, dims=(1,2))
        return y
    def _load_one(self, idx: int) -> torch.Tensor:
        it = self.data_items[idx]
        arr = load_5_band_image_from_parts(Path(it.class_path), it.base_name)
        if arr is None: raise RuntimeError(f"Load failed for {it.class_path}/{it.base_name}")
        return torch.from_numpy(arr.copy())
    
    def prewarm_cache(self):
        rk = int(os.environ.get("RANK", "0")); my_indices = list(range(len(self)))[rk::_torch_world_size_safe()]
        total_to_load = len(my_indices); rank_zero_print(f"[cache] Rank {rk}: Warming and pre-resizing {total_to_load} items...")
        for i, idx in enumerate(my_indices):
            try:
                if idx not in self.cache:
                    tensor_41 = self._load_one(idx)
                    self.cache[idx] = self.resize(tensor_41)
                if i > 0 and i % 2000 == 0 and is_global_zero_env(): 
                    print(f"  > Warmed {i}/{total_to_load} on Rank 0...", flush=True)
            except Exception as e:
                if i % 2000 == 0: rank_zero_print(f"[cache-warn] Rank {rk} error on idx {idx}: {e}")
        rank_zero_print(f"[cache] Rank {rk}: COMPLETED warming {len(self.cache)} items.")
        
    def __getitem__(self, idx: int):
        if idx in self.cache:
            x = self.cache[idx]
        else:
            x = self.resize(self._load_one(idx))
        return self._maybe_augment(x), self.labels[idx]

class LensDM(L.LightningDataModule):
    def __init__(self, data_items: List[Item], image_size: int, **hparams):
        super().__init__(); self.data_items = data_items; self.image_size = image_size
        self.save_hyperparameters(hparams)
    def setup(self, stage: Optional[str] = None):
        train_items, val_items = stratified_split(self.data_items, self.hparams.val_split, self.hparams.seed)
        self.ds_train = LensDataset(train_items, augment=True, image_size=self.image_size)
        self.ds_val = LensDataset(val_items, augment=False, image_size=self.image_size)
        if stage == "fit" and self.hparams.get("cache_ratio", 0.0) >= 1.0:
            rank_zero_print("Pre-warming 100% of cache..."); self.ds_train.prewarm_cache(); self.ds_val.prewarm_cache()
    def train_dataloader(self):
        ws = _torch_world_size_safe(); sampler = DistributedSampler(self.ds_train, shuffle=True, seed=self.hparams.seed) if ws > 1 else None
        return DataLoader(self.ds_train, batch_size=self.hparams.batch_size, shuffle=(ws==1), sampler=sampler, num_workers=self.hparams.num_workers, 
                          prefetch_factor=self.hparams.prefetch_factor, pin_memory=self.hparams.pin_memory, persistent_workers=True, drop_last=True)
    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          prefetch_factor=self.hparams.prefetch_factor, pin_memory=self.hparams.pin_memory, persistent_workers=True)

# --- Model & Training Logic ---
class LensClassifier(L.LightningModule):
    def __init__(self, **hparams):
        super().__init__(); self.save_hyperparameters()
        if _HAS_TIMM:
            create_kwargs = {'pretrained': True, 'in_chans': 5, 'num_classes': 1}
            if 'vit' in self.hparams.model_name.lower():
                create_kwargs['img_size'] = self.hparams.get('image_size', 41)
            self.model = timm.create_model(self.hparams.model_name, **create_kwargs)
            rank_zero_print(f"[model] Using timm backbone '{self.hparams.model_name}'")
        else:
            base = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1")
            base.features[0][0] = nn.Conv2d(5, 96, kernel_size=4, stride=4); base.classifier[2] = nn.Linear(base.classifier[2].in_features, 1)
            self.model = base; rank_zero_print("[model] Using fallback torchvision convnext_tiny")
        self.train_acc = BinaryAccuracy(); self.val_acc = BinaryAccuracy(); self.val_auc = BinaryAUROC()
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.model(x)
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.estimated_stepping_batches)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}
    def training_step(self, batch, batch_idx):
        x, y = batch; logits = self(x).squeeze(1); loss = F.binary_cross_entropy_with_logits(logits, y)
        self.train_acc.update(torch.sigmoid(logits), y.int())
        self.log_dict({"train_loss": loss, "train_acc": self.train_acc}, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch; logits = self(x).squeeze(1); probs = torch.sigmoid(logits); loss = F.binary_cross_entropy_with_logits(logits, y)
        self.val_acc.update(probs, y.int()); self.val_auc.update(probs, y.int())
        self.log_dict({"val_loss": loss, "val_acc": self.val_acc, "val_auc": self.val_auc}, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

# --- Main Execution ---
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["train"], default="train")
    p.add_argument("--train-roots", nargs="+", default=[]); p.add_argument("--train-roots-neg", nargs="+", default=[])
    p.add_argument("--val-split", type=float, default=0.1); p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--model-name", default="convnext_tiny"); p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8); p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--cache-ratio", type=float, default=1.0); p.add_argument("--index-cache", default="./index.pkl")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(); set_seed(args.seed)

    image_size = 224 if 'vit' in args.model_name else 41
    rank_zero_print(f"Using image size: {image_size}x{image_size}")

    cache_path, lock_path = Path(args.index_cache), Path(args.index_cache + ".lock")
    if is_global_zero_env() and not cache_path.exists():
        try:
            lock_path.touch(exist_ok=False); atexit.register(lambda p: p.unlink(missing_ok=True), lock_path)
            pos_items = build_index(args.train_roots, 1); neg_items = build_index(args.train_roots_neg, 0)
            with open(cache_path, "wb") as f: pickle.dump(pos_items + neg_items, f)
            lock_path.unlink()
        except FileExistsError:
            while lock_path.exists(): time.sleep(1)
    while not cache_path.exists(): time.sleep(1)
    with open(cache_path, "rb") as f: data_items = pickle.load(f)
    
    dm_kwargs = vars(args).copy(); dm_kwargs.pop('train_roots', None); dm_kwargs.pop('train_roots_neg', None)
    dm = LensDM(data_items=data_items, image_size=image_size, **dm_kwargs)

    model_kwargs = vars(args).copy(); model_kwargs['image_size'] = image_size
    model = LensClassifier(**model_kwargs)
    out_dir = Path("./runs") / args.model_name
    ckpt_cb = ModelCheckpoint(dirpath=out_dir / "ckpts", filename="{epoch}-{val_auc:.4f}", monitor="val_auc", mode="max", save_top_k=3)
    
    # Detect number of nodes from environment (torchrun sets these)
    num_nodes = int(os.environ.get("WORLD_SIZE", "1")) // int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    devices_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", "4"))
    
    trainer = L.Trainer(
        default_root_dir=out_dir, 
        max_epochs=args.epochs, 
        accelerator="gpu", 
        devices=devices_per_node,
        num_nodes=num_nodes,
        strategy=DDPStrategy(find_unused_parameters=False), 
        logger=CSVLogger(out_dir, name="logs"), 
        callbacks=[ckpt_cb, EarlyStopping("val_auc", mode="max", patience=8)],
        precision="16-mixed", 
        log_every_n_steps=20, 
        num_sanity_val_steps=0, 
        enable_model_summary=False, 
        enable_progress_bar=True
    )
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high'); main()
