# Interpreting MobileNetV3-Small on CIFAR-10

A computer-vision interpretability project studying three MobileNetV3-Small
variants — `vanilla` (standard architecture), `no_se` (squeeze-and-excitation
blocks removed), and `small_kernel` (5x5 depthwise convolutions replaced with
3x3) — trained on CIFAR-10. Explanations are produced with a from-scratch
Grad-CAM implementation and checked for faithfulness with the Adebayo et al.
(2018) cascading parameter-randomization sanity test. This is the foundation
for both a portfolio demo and a research paper on how architectural choices
affect the reliability of saliency-based explanations.

## Status

- **Phase 1 (done, `b7c0106`)** — Scaffold, data pipeline, MobileNetV3
  variants. Deterministic seeding, YAML configs with dot-access loading, the
  CIFAR-10 pipeline (32px augmentation, upsample to 224x224, ImageNet
  normalization), and the three model variants.
- **Phase 2 (done, `9cef2ee`)** — Training + evaluation harness. SGD +
  cosine-annealed training loop, checkpointing, per-class accuracy and
  confusion-matrix evaluation, and correct/incorrect prediction index
  artifacts for downstream explanation work.
- **Phase 3 (done, `3e2edde`)** — From-scratch Grad-CAM module. Hook-based
  Grad-CAM (Selvaraju et al., 2017) implementation and overlay visualization,
  independent of any third-party CAM library.
- **Phase 4 (done, `5bcaeff`)** — Grad-CAM sanity checks. Cascading
  (top-down) model-parameter randomization test with Spearman and SSIM
  similarity metrics, quantifying whether Grad-CAM explanations actually
  track learned weights rather than acting as edge detectors.
- **Phase 5+ (planned)** — Cross-variant ablation comparison (vanilla vs.
  no-SE vs. small-kernel explanation quality), robustness checks, a demo
  app, and the write-up for the research paper.

## Results so far

- `vanilla_scratch` reaches **~80.5% test accuracy** on CIFAR-10.
- Grad-CAM passes the cascading parameter-randomization sanity check:
  similarity to the original CAM decays from ~0.8 toward ~0 as the model is
  progressively randomized top-down, confirming the explanations are
  weight-dependent rather than degenerate edge maps.

## Project layout

```
configs/                              One YAML per experiment.
src/utils/seed.py                     Deterministic seeding (Python/NumPy/torch/cudnn).
src/utils/config.py                   YAML -> dot-access config loader.
src/data/cifar10.py                   Loaders, transforms, denormalize, class names.
src/models/mobilenetv3_variants.py    The three MobileNetV3-Small architectures.
src/train/engine.py                   Train/eval loops, checkpointing, eval artifacts.
src/explain/gradcam.py                From-scratch Grad-CAM + overlay visualization.
src/explain/sanity.py                 Cascading parameter-randomization sanity check.
scripts/train.py                      Train a variant from a config.
scripts/gradcam_demo.py               Save Grad-CAM overlay panels for a checkpoint.
scripts/sanity_check.py               Run the Grad-CAM sanity check on a checkpoint.
scripts/smoke_test*.py                Fast, no-download checks for each module.
```

## Quick Start

```powershell
# Install dependencies
pip install -r requirements.txt

# Train a variant (downloads CIFAR-10 on first run)
python scripts/train.py --config configs/vanilla_scratch.yaml

# Grad-CAM overlay panels for a trained checkpoint
python scripts/gradcam_demo.py --checkpoint runs/vanilla_scratch/checkpoints/best.pth

# Cascading parameter-randomization sanity check
python scripts/sanity_check.py --checkpoint runs/vanilla_scratch/checkpoints/best.pth
```

## Smoke tests

Each module has a smoke test that runs without downloading CIFAR-10:

```powershell
python scripts/smoke_test.py           # data/config/model utilities
python scripts/smoke_test_train.py     # training + evaluation harness
python scripts/smoke_test_gradcam.py   # Grad-CAM
python scripts/smoke_test_sanity.py    # sanity-check mechanics
```

## Configs

Each experiment config (`configs/*.yaml`) defines `model`, `data`, and
`train` sections. Load one with:

```python
from src.utils import load_config

cfg = load_config("configs/vanilla_scratch.yaml")
cfg.model.variant  # dot-access
```
