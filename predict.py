#!/usr/bin/env python3
import os, argparse, re, importlib.util
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# --- Dynamic Import for t.py ---
def import_train_script():
    candidates = ["t.py", "strong-lens-mi300-train_infer.py", "train_v2.py"]
    for c in candidates:
        if Path(c).exists():
            spec = importlib.util.spec_from_file_location("train_module", c)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(f"Could not find training script. Looked for: {candidates}")

# Import the core classes from your training script
train_script = import_train_script()
LensClassifier = train_script.LensClassifier
LensDataset = train_script.LensDataset
build_index = train_script.build_index
print(f"Successfully imported classes from {train_script.__file__}")

# --- Helper Functions ---
def collate_fn(batch):
    # Filter out None values from dataset loading errors
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

@torch.no_grad()
def predict_csv(models, data_items, batch_size, num_workers, out_csv, image_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for m in models:
        m.to(device).eval()
    
    # --- FIX: Match t.py signature (augment=False, image_size=...) ---
    # t.py LensDataset does NOT take 'transforms', it takes 'augment' and 'image_size'
    ds = LensDataset(data_items, augment=False, image_size=image_size)

    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    all_probs = []
    print(f"Starting prediction on {len(data_items)} items...")
    
    count = 0
    for batch in dl:
        if batch is None or batch[0] is None: continue
        
        x, _ = batch
        x = x.to(device)
        
        # Ensemble / TTA Logic (Original + HFlip + VFlip)
        model_logits = []
        for m in models:
            logits_orig = m(x).squeeze(1)
            logits_h = m(torch.flip(x, dims=(3,))).squeeze(1)
            logits_v = m(torch.flip(x, dims=(2,))).squeeze(1)
            
            # Average TTA for this model
            avg_logits = torch.stack([logits_orig, logits_h, logits_v]).mean(0)
            model_logits.append(avg_logits)
        
        # Average across all models in ensemble
        ensemble_logits = torch.stack(model_logits).mean(0)
        probs = torch.sigmoid(ensemble_logits)
        all_probs.append(probs.cpu())

        count += 1
        if count % 20 == 0:
            print(f"  Processed batch {count}/{len(dl)}...", end='\r')

    if not all_probs:
        print("\nError: No predictions generated. Check dataset paths.")
        return

    final_probs = torch.cat(all_probs).numpy()
    
    # Write to CSV
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("id,pred\n")
        for it, p in zip(data_items, final_probs):
            f.write(f"{it.obj_id},{p:.6f}\n")
            
    print(f"\nSuccess! Wrote {len(data_items)} predictions to -> {out_path}")

def find_best_checkpoints(run_dir: Path):
    ckpt_dir = run_dir / "ckpts"
    if not ckpt_dir.is_dir(): return []
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    scores = []
    for p in ckpts:
        match = re.search(r"val_auc=(\d+\.\d+)", p.name)
        scores.append(float(match.group(1)) if match else -1.0)
    sorted_pairs = sorted(zip(scores, ckpts), key=lambda x: x[0], reverse=True)
    return [p for s, p in sorted_pairs]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--out-csv", default="./final_submission.csv")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    all_best_ckpts = []
    model_names = []
    for d in args.run_dirs:
        path = Path(d)
        best = find_best_checkpoints(path)
        if best:
            print(f"Found checkpoint in {d}: {best[0].name}")
            all_best_ckpts.append(best[0])
            model_names.append(path.name)
        else:
            print(f"WARNING: No checkpoints found in {d}")

    if not all_best_ckpts:
        print("No models found. Exiting.")
        return

    image_size = 224 if any('vit' in n for n in model_names) else 41
    print(f"Using image size: {image_size}x{image_size}")

    data_items = build_index([args.test_root], -1)
    if not data_items:
        print(f"ERROR: No items found in {args.test_root}")
        return

    models = [LensClassifier.load_from_checkpoint(p) for p in all_best_ckpts]
    predict_csv(models, data_items, args.batch_size, args.num_workers, args.out_csv, image_size)

if __name__ == "__main__":
    main()
