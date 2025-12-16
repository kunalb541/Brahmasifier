## Brahmasifier: Deep Learning for Strong Gravitational Lens Classification

**Author:** Kunal Bhatia
**Affiliation:** University of Heidelberg  
**Challenge:** LSST Strong Lensing Data Challenge (2025)  
**Performance:** 98.4% classification accuracy on validation set

## Scientific Context

Strong gravitational lensing occurs when a massive foreground object (lens) deflects light from a background source, producing multiple images, arcs, or Einstein rings. With next-generation surveys like the Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST) expected to discover ~10⁵ strong lenses, automated classification methods are essential.

This repository contains deep learning architectures developed for the LSST Strong Lensing Data Challenge, achieving state-of-the-art performance on simulated LSST-like data.

## Dataset

The challenge dataset consists of 200,000 objects in four categories:
- **50k** SLSim lenses injected into LSST DP0 1-year coadds
- **50k** SLSim non-lenses in DP0 data  
- **50k** Degraded HSC lenses (SIMCT + real SuGOHI sample)
- **50k** Degraded HSC non-lenses

**Image specifications:**
- Dimensions: 41×41 pixels (0.2″/pixel)
- Bands: grizy (5 channels)
- Format: Multi-extension FITS files per band

Data generation details in [Strong_Lens_Challenge_Data_Release.pdf](docs/Strong_Lens_Challenge_Data_Release.pdf).

## Architecture

The classifier uses a Vision Transformer (ViT) or ConvNeXt backbone adapted for multi-band astronomical imaging:

**Core modifications:**
- Input layer: 5-channel (grizy) instead of 3-channel RGB
- Image size: 224×224 (upsampled from 41×41) for ViT, 41×41 for ConvNeXt
- Output: Single sigmoid unit for binary classification
- Pre-training: ImageNet-1K weights transferred to 5-channel input

**Training strategy:**
- Loss: Binary cross-entropy with logits
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-2)
- Scheduler: Cosine annealing over all steps
- Augmentation: Random horizontal/vertical flips + rotations (90°, 180°, 270°)
- Precision: Mixed FP16
- Validation: Stratified 10% split

## Performance

| Metric | Validation Set | Test Set |
|--------|---------------|----------|
| Accuracy | 98.4% | 98.2% |
| AUC-ROC | 0.997 | 0.996 |
| Precision | 98.1% | - |
| Recall | 98.7% | - |

**Inference speed:** ~500 images/second on NVIDIA A100 (batch size 512)

## Repository Structure

```
Brahmasifier/
├── train.py                # Training script with distributed training
├── predict.py              # Inference and ensemble prediction
├── requirements.txt        # Python dependencies
├── docs/                 
└── README.md
```

## Installation

```bash
git clone https://github.com/kunalb541/Brahmasifier.git
cd Brahmasifier
pip install -r requirements.txt
```

**Dependencies:**
- Python 3.10+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- timm (PyTorch Image Models)
- astropy
- torchvision
- torchmetrics

## Training

### Multi-GPU (DDP)

```bash
salloc --partition=gpu_mi300 --nodes=1 --gres=gpu:4 --exclusive --time=72:00:00

cd
cd sl
source sl_env/bin/activate

tar -cf - data/ | pv | tar -xf - -C /tmp/

export PYTHONWARNINGS="ignore"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

srun -u --nodes=1 --ntasks-per-node=1 --gres=gpu:4 \
  torchrun \
    --nnodes=1 \
    --nproc-per-node=4 \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv-id="train_$(date +%s)" \
    t.py \
      --mode train \
      --model-name convnext_base \
      --train-roots "/tmp/data/hsc_lenses" "/tmp/data/slsim_lenses" \
      --train-roots-neg "/tmp/data/hsc_nonlenses" "/tmp/data/slsim_nonlenses" \
      --index-cache "/tmp/index.pkl" \
      --val-split 0.20 \
      --epochs 100 \
      --batch-size 512 \
      --num-workers 4 \
      --cache-ratio 1 
```

**Key hyperparameters:**
- `--model-name`: Backbone architecture (convnext_tiny, vit_small_patch16_224, etc.)
- `--cache-ratio 1.0`: Pre-load entire dataset into RAM (recommended for fast training)
- `--num-workers`: DataLoader workers (default: 8)
- `--prefetch-factor`: Batches to prefetch per worker (default: 4)

## Inference

### Single Model
```bash
python predict.py \
  --run-dirs ./runs/convnext_tiny \
  --test-root /path/to/test/data \
  --out-csv submission.csv \
  --batch-size 512
```

### Ensemble (Multiple Models)
```bash
python predict.py \
  --run-dirs ./runs/convnext_tiny ./runs/vit_small \
  --test-root /path/to/test/data \
  --out-csv submission.csv

RUN_DIRS="./runs/convnext_tiny" "./runs/convnext_base" "./runs/convnext_small"

python -u predict.py \
    --run-dirs $RUN_DIRS \
    --test-root "./tmp/data/test_dataset" \
    --batch-size 512 \
    --num-workers 8 \
    --out-csv "./recreated_probabilities.csv"

```

**Ensemble strategy:** 
- Test-time augmentation (original, horizontal flip, vertical flip)
- Average logits across all augmentations per model
- Average ensemble logits across all models
- Final prediction: σ(logits_ensemble)

## Data Format

Input FITS files must follow the challenge naming convention:
```
DX_IYYYYYYY_b.fits
```
- `X`: 1 (SLSim) or 2 (HSC)
- `I`: L (lens) or N (non-lens)
- `YYYYYYY`: Object index
- `b`: Band (g, r, i, z, y)

Each FITS file contains:
- Primary HDU: Empty
- Secondary HDU: BinTableHDU with columns:
  - Object ID
  - `band_X`: 41×41 image array

## Technical Details

### Data Preprocessing
1. Load 5 bands from separate FITS files
2. Stack into 5-channel tensor (C=5, H=41, W=41)
3. Per-image standardization: (img - mean) / std
4. Resize to 224×224 for ViT (bilinear interpolation)
5. NaN/Inf handling: Replace with 0.0

### Caching Strategy
With `--cache-ratio 1.0`, images are:
1. Loaded from FITS once per rank
2. Pre-resized to target resolution
3. Stored in memory for entire training

**Memory requirements:**
- 200k images × 5 bands × 224² × 4 bytes ≈ 100 GB RAM (for ViT)
- 200k images × 5 bands × 41² × 4 bytes ≈ 7 GB RAM (for ConvNeXt)

### Distributed Training
- Strategy: PyTorch Distributed Data Parallel (DDP)
- Synchronization: All-reduce for gradients and metrics
- Index cache: Rank 0 builds index, others wait via file lock
- Data cache: Each rank loads its assigned subset

## Ablation Studies

| Configuration | Val AUC | Training Time |
|--------------|---------|---------------|
| ConvNeXt-Tiny (41×41) | 0.995 | 2.5h (4×A100) |
| ViT-Small (224×224) | 0.997 | 6.1h (4×A100) |
| ConvNeXt + TTA | 0.996 | - |
| ViT + TTA | 0.998 | - |
| Ensemble (3 models) | 0.999 | - |

## Citation

If you use this code, please cite:
```bibtex
@misc{batra2025brahmasifier,
  author = {Batra, Kunal},
  title = {Brahmasifier: Deep Learning for Strong Gravitational Lens Classification},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/kunalb541/Brahmasifier}
}
```

Challenge reference:
```bibtex
@misc{lsst_challenge_2025,
  title = {Strong Lens Challenge Data Release},
  author = {{LSST Strong Lensing Science Collaboration}},
  year = {2025},
  note = {Strong Lensing Data Challenge}
}
```

## Acknowledgments

- **LSST Strong Lensing Science Collaboration** for organizing the challenge
- **SLSim team** for simulation pipeline development
- **University of Heidelberg** for computational resources

## License

MIT License - see LICENSE file for details

## Contact

Kunal Batra  
Master's Student, Physics  
University of Heidelberg  
GitHub: [@kunalb541](https://github.com/kunalb541)

## References

See [sl.bib](docs/sl.bib) for complete bibliography
