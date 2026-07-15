# Interpreting MobileNetV3-Small on CIFAR-10

A computer-vision interpretability project studying MobileNetV3-Small trained
on CIFAR-10 (images upsampled from 32x32 to 224x224), with an architectural
ablation over the squeeze-and-excitation blocks and depthwise kernel size.

## Phase 1 (this phase): foundation

- Deterministic seeding utilities.
- YAML-based experiment configs with dot-access loading.
- CIFAR-10 data pipeline: augmentation happens at native 32px resolution
  before the image is upsampled to 224x224; ImageNet normalization statistics
  are used throughout since the model architecture originates from ImageNet
  pretraining.
- Three MobileNetV3-Small architecture variants for the ablation:
  - `vanilla`: standard MobileNetV3-Small (supports ImageNet-pretrained weights).
  - `no_se`: squeeze-and-excitation blocks removed from every stage.
  - `small_kernel`: all 5x5 depthwise convolutions replaced with 3x3.

No training loop or attribution methods (Grad-CAM, LIME, etc.) are
implemented yet — those arrive in later phases.

## Project layout

```
configs/                     One YAML per experiment.
src/utils/seed.py            Deterministic seeding (Python/NumPy/torch/cudnn).
src/utils/config.py          YAML -> dot-access config loader.
src/data/cifar10.py          Loaders, transforms, denormalize, class names.
src/models/mobilenetv3_variants.py   The three MobileNetV3-Small architectures.
scripts/smoke_test.py        End-to-end sanity check (no dataset download).
```

## Setup

```powershell
pip install -r requirements.txt
```

## Smoke test

Verifies the model variants, data utilities, and config loader without
downloading CIFAR-10:

```powershell
python scripts/smoke_test.py
```

## Configs

Each experiment config (`configs/*.yaml`) defines `model`, `data`, and
`train` sections. Load one with:

```python
from src.utils import load_config

cfg = load_config("configs/vanilla_scratch.yaml")
cfg.model.variant  # dot-access
```
