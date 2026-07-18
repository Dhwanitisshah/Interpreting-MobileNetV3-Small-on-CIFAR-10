# Model weights

Trained checkpoints (`*.pth`) are **not** committed to this repository — they
are reproducible in a few minutes to a few hours per variant on a single GPU
(or, more slowly, on CPU) and would otherwise bloat the git history.

## Where checkpoints live

Every training run writes its checkpoints under `runs/<experiment>/checkpoints/`,
where `<experiment>` matches the `experiment` field in the corresponding
`configs/*.yaml` file:

```
runs/
  vanilla_scratch/checkpoints/{best,last}.pth
  no_se_scratch/checkpoints/{best,last}.pth
  small_kernel_scratch/checkpoints/{best,last}.pth
  vanilla_finetune/checkpoints/{best,last}.pth
```

`best.pth` is the checkpoint with the highest validation accuracy seen during
training; `last.pth` is the final epoch. Downstream scripts (Grad-CAM,
faithfulness, robustness, comparison) default to `best.pth`.

## Reproducing them

```powershell
python scripts/train.py --config configs/vanilla_scratch.yaml
python scripts/train.py --config configs/no_se_scratch.yaml
python scripts/train.py --config configs/small_kernel_scratch.yaml
python scripts/train.py --config configs/vanilla_finetune.yaml
```

Each run downloads CIFAR-10 on first use, trains with the seed fixed in its
config, and reports final test accuracy plus per-class accuracy (see the main
[README](../README.md#results-so-far) for the numbers this codebase produces).

## Pretrained release

No pretrained checkpoints are hosted externally at this time. If/when a
release is published (e.g. via GitHub Releases or a model-hosting service),
the download links will be added here.
