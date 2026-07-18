# Interpreting MobileNetV3-Small on CIFAR-10

**Do architectural choices change how much you can trust a saliency map?**

A from-scratch interpretability study of three MobileNetV3-Small variants —
`vanilla` (standard architecture), `no_se` (squeeze-and-excitation blocks
removed), and `small_kernel` (5x5 depthwise convolutions replaced with 3x3) —
trained on CIFAR-10. Explanations are produced with a from-scratch Grad-CAM
implementation and evaluated along three independent axes: sanity
(parameter-randomization), faithfulness (deletion/insertion/ROAD), and
robustness to distribution shift (explanation drift under corruption). This
is the codebase behind an in-progress research paper on how architectural
choices affect the reliability of saliency-based explanations.

## Research motivation

Grad-CAM and similar saliency methods are widely used to explain CNN
predictions, but two failure modes are well documented in the literature:
explanations can be insensitive to the model's learned weights (Adebayo et
al., 2018), and they are not guaranteed to reflect what the model actually
relies on to make a prediction (i.e., they can fail faithfulness checks).
This project asks a narrower, less-studied question: **holding the task and
training procedure fixed, does changing the architecture change the
reliability of its explanations?** Three matched MobileNetV3-Small variants
are compared so that any measured difference is attributable to one
architectural change (removing SE blocks, or shrinking the depthwise kernel)
rather than to training procedure, dataset, or capacity differences.

## Features

- **From-scratch Grad-CAM** (Selvaraju et al., 2017) — hook-based
  implementation with overlay visualization, independent of any third-party
  CAM library.
- **Sanity checks** — the cascading (top-down) parameter-randomization test
  (Adebayo et al., 2018), quantified with Spearman and SSIM similarity to
  the original CAM, to confirm explanations are weight-dependent rather than
  degenerate edge detectors.
- **Quantitative faithfulness** — deletion/insertion AUC and ROAD
  (Remove-and-Debias, Rong et al., 2022) gap, with confidence-normalization
  (dividing by the model's own initial confidence) so faithfulness is
  comparable across models with different calibration, plus paired
  significance testing (t-test, Wilcoxon, Cohen's d, Bonferroni correction)
  and TOST equivalence testing.
- **Cross-variant Grad-CAM comparison** — the same test images run through
  multiple checkpoints, rendered side by side.
- **Distribution-shift robustness** — Grad-CAM drift under six
  ImageNet-C-style corruptions at three severities, measured against a
  *fixed* target class (the clean prediction) so drift isn't conflated with
  the model simply answering a different question after corruption; a
  CAM-concentration diagnostic and per-corruption breakdown rule out
  confounds and localize the effect.
- **Fully reproducible** — every experiment is defined by a YAML config with
  a fixed seed; every metric has a corresponding smoke test that runs
  without downloading CIFAR-10.

## Results so far

- `vanilla_scratch` reaches **~80.5% test accuracy** on CIFAR-10 after 10
  epochs of from-scratch training.
- Grad-CAM passes the cascading parameter-randomization sanity check:
  similarity to the original CAM decays from ~0.8 toward ~0 as the model is
  progressively randomized top-down, confirming the explanations are
  weight-dependent rather than degenerate edge maps.
- Faithfulness (deletion/insertion AUC, ROAD gap), once confidence-normalized,
  is a **formal TOST equivalence** (SESOI = 0.3 Cohen's d) between
  `vanilla_scratch` and `no_se_scratch` on all three metrics — removing SE
  blocks does not measurably change *static* faithfulness. Other pairs and a
  small number of exceptions are detailed in `runs/faithfulness/report.txt`.
- Under distribution shift, explanation drift scales with architecture beyond
  what accuracy loss alone predicts, but the effect is **directional, not
  uniform across corruptions** — see the results table and caveats below.
- The SE result extends to drift stability: 3 of 4 drift metrics (Spearman,
  SSIM, top-k IoU) are formally **equivalent** between `vanilla_scratch` and
  `no_se_scratch` at the same SESOI; the fourth (centroid shift) is
  **inconclusive** (p_TOST = 0.055, just above the 0.05 threshold) rather
  than equivalent — likely underpowered, since its observed difference
  (spearman: mean diff +0.0030) sits within the ~0.003 run-to-run
  measurement-noise floor established below.

See [`CHANGELOG.md`](CHANGELOG.md) for the full phase-by-phase development
history, including the Phase 7.5 evaluation-reproducibility fix these numbers
depend on.

### Headline results table

Accuracy-floor-filtered (`--min-accuracy 0.2`): drift/accuracy cells where any
model's accuracy under corruption falls to or near chance (10-class chance =
0.100) are excluded, since predictions there are uninterpretable and inflate
apparent drift. Excluded cells: `gaussian_noise` at severities 1/3/5 and
`contrast` at severity 5. Unfiltered numbers (all cells) are shown alongside
for reference.

| Model | Test accuracy | Norm. deletion AUC ↓ | Norm. insertion AUC ↑ | Norm. ROAD gap ↑ | Mean drift, filtered (1 − Spearman) | Drift/acc-drop ratio, filtered | Mean drift, unfiltered | Drift/acc-drop ratio, unfiltered |
|---|---|---|---|---|---|---|---|---|
| `vanilla_scratch` | 80.55% | 0.288 | 0.833 | 0.454 | 0.107 | 0.69 | 0.222 | 0.83 |
| `no_se_scratch` | 81.29% | 0.281 | 0.825 | 0.504 | 0.097 | 0.71 | 0.219 | 0.85 |
| `small_kernel_scratch` | 76.34% | 0.269 | 0.878 | 0.540 | 0.141 | 0.99 | 0.262 | 1.07 |
| `vanilla_finetune` | 95.59% | 0.268 | 0.781 | 0.551 | 0.159 | 1.26 | 0.333 | 1.20 |

Lower is more faithful for deletion AUC; higher is more faithful for
insertion AUC and ROAD gap. Drift and drift/acc-drop ratios are averaged over
all six corruptions and three severities (minus the excluded cells, for the
filtered columns). Source: `runs/faithfulness/report.txt` and
`runs/robustness_fixed_seeded/report.txt` (see Reproducibility notes for why
this directory, not `runs/robustness/`, is now canonical).

**On the pre-training (`vanilla_finetune`) result — directional, not
general.** Per-corruption excess drift (`vanilla_finetune` minus the other
three models' mean, unfiltered) is elevated on `motion_blur` (+0.084),
`jpeg_compression` (+0.164), and `defocus_blur` (+0.100); **negligible** on
`brightness` (+0.014); and **reversed** on `contrast` (−0.007 unfiltered,
−0.145 filtered — `vanilla_finetune` drifts *less* than the other models
there). `gaussian_noise` shows the largest excess (+0.317) but every one of
its severities is in the excluded, near-chance-accuracy set, so it is not
part of the reliable claim. The honest summary is that ImageNet pretraining
increases explanation drift on some corruption families and decreases it on
at least one, not that it uniformly destabilizes explanations.

## Repository structure

```
configs/                   One YAML per experiment (seed, model variant, data, training hyperparameters).
src/
  data/cifar10.py             CIFAR-10 loaders, transforms, denormalization, class names.
  models/mobilenetv3_variants.py  The three MobileNetV3-Small architectures.
  train/engine.py             Train/eval loops, checkpointing, eval artifacts.
  explain/gradcam.py           From-scratch Grad-CAM + overlay visualization.
  explain/sanity.py            Cascading parameter-randomization sanity check.
  explain/compare.py           Cross-variant Grad-CAM comparison.
  metrics/faithfulness.py      Deletion/insertion AUC, ROAD gap, paired significance testing.
  metrics/equivalence.py       TOST equivalence testing (used across faithfulness/robustness reports).
  robustness/corruptions.py    ImageNet-C-style corruption functions.
  robustness/drift.py          Grad-CAM drift measurement under corruption.
  utils/                       Config loading, seeding, and shared script helpers.
scripts/
  train.py                     Train a variant from a config.
  gradcam_demo.py               Save Grad-CAM overlay panels for a checkpoint.
  sanity_check.py               Run the Grad-CAM sanity check on a checkpoint.
  compare_variants.py           Run the cross-variant Grad-CAM comparison.
  faithfulness_eval.py          Run faithfulness metrics on a set of checkpoints.
  report_faithfulness.py        Reporting layer over faithfulness runs (summaries, TOST, p0 diagnostic).
  robustness_eval.py            Run the robustness/drift evaluation across checkpoints.
  report_robustness.py          Per-corruption drift breakdown, accuracy-floor sensitivity analysis.
  concentration_diagnostic.py   CAM concentration diagnostic + drift equivalence testing.
  smoke_test*.py                Fast, no-download checks for each module.
runs/                       Experiment outputs (checkpoints, metrics, figures) — gitignored, see below.
weights/README.md          Where to get / how to regenerate trained checkpoints.
```

## Installation

Requires Python 3.10+ and PyTorch 2.2+. Developed and tested with
Python 3.13, PyTorch 2.11 (CPU build); a CUDA-enabled PyTorch build will be
used automatically if available (`--device auto`, the default).

```powershell
git clone <this-repo>
cd RESEARCH_PAPER
pip install -r requirements.txt
```

## Dataset preparation

No manual download is required. `scripts/train.py` and every evaluation
script download CIFAR-10 via `torchvision.datasets.CIFAR10` into `./data` on
first use (`--data-root` to override). Every script also accepts
`--no-download`, which substitutes a synthetic random dataset — useful for
smoke-testing a full pipeline without a network connection.

## Training

```powershell
python scripts/train.py --config configs/vanilla_scratch.yaml
python scripts/train.py --config configs/no_se_scratch.yaml
python scripts/train.py --config configs/small_kernel_scratch.yaml
```

Each config defines `model`, `data`, and `train` sections and is loaded with
dot-access:

```python
from src.utils import load_config

cfg = load_config("configs/vanilla_scratch.yaml")
cfg.model.variant  # "vanilla"
```

Outputs go to `runs/<experiment>/` (checkpoints, `metrics.json`, and eval
artifacts — see [`weights/README.md`](weights/README.md) for the checkpoint
layout).

## Fine-tuning

`configs/vanilla_finetune.yaml` starts from ImageNet-pretrained weights
(`model.pretrained: true`, only valid for the `vanilla` variant, since the
`no_se`/`small_kernel` ablations change the network topology and can't
reuse ImageNet weights):

```powershell
python scripts/train.py --config configs/vanilla_finetune.yaml
```

## Evaluation

Training automatically runs a final evaluation pass and writes
`runs/<experiment>/eval/{predictions,confusion_matrix,correct_indices,incorrect_indices}.json`.
These per-image records are what downstream Grad-CAM/faithfulness scripts
use to pick correctly-classified visualization images.

## Grad-CAM

```powershell
python scripts/gradcam_demo.py --checkpoint runs/vanilla_scratch/checkpoints/best.pth --num-images 8
```

Saves original/heatmap/overlay panels to `runs/gradcam_demo/`.

## Ablation study

Cross-variant Grad-CAM comparison — same test images, multiple checkpoints,
side by side:

```powershell
python scripts/compare_variants.py --checkpoints `
    runs/vanilla_scratch/checkpoints/best.pth `
    runs/no_se_scratch/checkpoints/best.pth `
    runs/small_kernel_scratch/checkpoints/best.pth --num-images 6 --only-all-correct
```

Quantitative faithfulness across the same variants:

```powershell
python scripts/faithfulness_eval.py --checkpoints `
    runs/vanilla_scratch/checkpoints/best.pth `
    runs/no_se_scratch/checkpoints/best.pth `
    runs/small_kernel_scratch/checkpoints/best.pth `
    runs/vanilla_finetune/checkpoints/best.pth --num-images 500

python scripts/report_faithfulness.py
```

## Distribution-shift experiments

```powershell
python scripts/robustness_eval.py --checkpoints `
    runs/vanilla_scratch/checkpoints/best.pth `
    runs/no_se_scratch/checkpoints/best.pth `
    runs/small_kernel_scratch/checkpoints/best.pth `
    runs/vanilla_finetune/checkpoints/best.pth `
    --num-images 200 --seed 42 --output-dir runs/robustness_fixed_seeded

python scripts/report_robustness.py --robustness-dir runs/robustness_fixed_seeded --min-accuracy 0.2
python scripts/concentration_diagnostic.py --checkpoints <same checkpoints as above> `
    --robustness-metrics runs/robustness_fixed_seeded/robustness_metrics.json `
    --output-dir runs/robustness_fixed_seeded --seed 42
```

`--output-dir`/`--robustness-dir` are shown explicitly here (rather than the
scripts' `runs/robustness` default) to avoid overwriting this repo's
canonical `runs/robustness_fixed_seeded/` run — see Reproducibility notes.

## Reliability verification (sanity checks)

Cascading parameter-randomization test (Adebayo et al., 2018) — confirms
Grad-CAM explanations track learned weights rather than acting as edge
detectors:

```powershell
python scripts/sanity_check.py --checkpoint runs/vanilla_scratch/checkpoints/best.pth
```

## Smoke tests

Each module has a smoke test that runs without downloading CIFAR-10:

```powershell
python scripts/smoke_test.py                 # data/config/model utilities
python scripts/smoke_test_train.py           # training + evaluation harness
python scripts/smoke_test_gradcam.py         # Grad-CAM
python scripts/smoke_test_sanity.py          # sanity-check mechanics
python scripts/smoke_test_compare.py         # cross-variant comparison
python scripts/smoke_test_faithfulness.py    # faithfulness metrics
python scripts/smoke_test_robustness.py      # corruptions + drift
```

## Example outputs

Running the commands above populates `runs/` with (a static, representative
copy of each figure below is committed under `docs/assets/` for this README;
`runs/` itself is gitignored and regenerated locally):

**Grad-CAM overlay** (`runs/gradcam_demo/gradcam_*.png`)

![Grad-CAM overlay example](docs/assets/gradcam_example.png)

**Sanity check decay** (`runs/sanity/<experiment>/decay_plot.png`) — Spearman/SSIM similarity to the original CAM as the model is progressively randomized top-down:

![Sanity check decay plot](docs/assets/sanity_decay_plot.png)

**Faithfulness curves and AUC comparison** (`runs/faithfulness/*.png`):

![Deletion/insertion curves](docs/assets/faithfulness_curves.png)
![Faithfulness AUC bar chart](docs/assets/faithfulness_auc_bars.png)

**Robustness under distribution shift** (`runs/robustness/*.png`):

![Drift vs. corruption severity](docs/assets/robustness_drift_vs_severity.png)
![Accuracy vs. corruption severity](docs/assets/robustness_accuracy_vs_severity.png)

Cross-variant comparison grids (`runs/compare/*.png`) are omitted here for
file size; regenerate them with `scripts/compare_variants.py`.

## Experiment pipeline

```mermaid
flowchart LR
    A[configs/*.yaml] --> B[scripts/train.py]
    B --> C["runs/&lt;experiment&gt;/checkpoints/*.pth"]
    C --> D[scripts/gradcam_demo.py]
    C --> E[scripts/sanity_check.py]
    C --> F[scripts/compare_variants.py]
    C --> G[scripts/faithfulness_eval.py]
    C --> H[scripts/robustness_eval.py]
    G --> I[scripts/report_faithfulness.py]
    H --> J[scripts/report_robustness.py]
    H --> K[scripts/concentration_diagnostic.py]
    D --> L[runs/gradcam_demo/*.png]
    E --> M["runs/sanity/&lt;experiment&gt;/*.png, *.json"]
    F --> N[runs/compare/*.png]
    I --> O[runs/faithfulness/report.txt]
    J --> P["runs/&lt;robustness-output-dir&gt;/report.txt"]
    K --> Q["runs/&lt;robustness-output-dir&gt;/concentration_report.txt"]
```

`<robustness-output-dir>` is whatever `--output-dir` you pass to
`robustness_eval.py`; this README's canonical numbers come from
`runs/robustness_fixed_seeded/` (see Reproducibility notes above) rather than
the older `runs/robustness/`.

## Reproducibility notes

- Every config fixes `seed: 42`; `src.utils.set_seed` seeds Python, NumPy,
  and torch (CPU + CUDA) and disables cuDNN benchmark-mode autotuning for
  deterministic kernels. As of Phase 7.5, every `scripts/*.py` entry point
  that evaluates a trained model (`robustness_eval.py`, `faithfulness_eval.py`,
  `sanity_check.py`, `concentration_diagnostic.py`) calls `set_seed(args.seed)`
  at the top of `main()`, before any data loading or evaluation.
- **Phase 7.5 fix and its effect on the robustness numbers.** Before this fix,
  `robustness_eval.py` seeded only the local image-selection RNG, leaving
  NumPy's *global* RNG state unseeded — and the `imagecorruptions` library
  draws its noise realizations (e.g. motion-blur angle, Gaussian noise) from
  that global state. Two runs with an identical `--seed` and identical
  checkpoints could therefore silently disagree: measured run-to-run
  variation was ~0.003 on mean drift values of ~0.22 (e.g. `no_se_scratch`:
  0.220 vs. 0.217; `small_kernel_scratch`: 0.258 vs. 0.261). This is why
  `runs/robustness_fixed_seeded/` — generated after the fix and verified
  below — rather than the older `runs/robustness/` or `runs/robustness_fixed/`,
  is the canonical source for every robustness number in this README.
  `runs/robustness/` and `runs/robustness_fixed/` are left as-is (not deleted)
  as a record of the pre-fix runs.
- **Verification.** `scripts/smoke_test_robustness.py` asserts *exact*
  (bitwise, not `np.allclose`) equality of every drift field across two
  in-process reruns of `evaluate_robustness` on synthetic data with the seed
  reset between them. On real data, two independent full runs
  (`runs/robustness_fixed_seeded/` and `runs/robustness_reprocheck/`, same
  checkpoints/`--num-images 200`/`--seed 42`) were diffed programmatically:
  all 57,600 drift-field values across 14,400 records were bitwise-identical.
- Checkpoints record their own `val_acc` and training `config`; evaluation
  scripts cross-check a loaded checkpoint's recorded `val_acc` against its
  experiment's `metrics.json` and fail loudly on a mismatch, so a
  stale/regenerated checkpoint can never silently produce misleading
  downstream numbers.
- Developed with Python 3.13 and PyTorch 2.11 (CPU); any PyTorch 2.2+ build
  (CPU or CUDA) should reproduce the same results modulo standard
  floating-point/GPU nondeterminism.

## Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{shah2026interpreting,
  title  = {Interpreting MobileNetV3-Small on CIFAR-10: Architectural
            Effects on Saliency Explanation Reliability},
  author = {Shah, Dhwanit},
  year   = {2026},
  howpublished = {\url{<repository-url>}}
}
```

## License

[MIT](LICENSE).

## Acknowledgements

- Grad-CAM: Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from
  Deep Networks via Gradient-based Localization." ICCV 2017.
- Sanity checks: Adebayo, J., et al. "Sanity Checks for Saliency Maps."
  NeurIPS 2018.
- ROAD: Rong, Y., et al. "A Consistent and Efficient Evaluation Strategy for
  Attribution Methods." ICML 2022.
- MobileNetV3: Howard, A., et al. "Searching for MobileNetV3." ICCV 2019.
  Base architecture and ImageNet-pretrained weights via `torchvision`.
- Corruption functions: the `imagecorruptions` package (Hendrycks & Dietterich,
  "Benchmarking Neural Network Robustness to Common Corruptions and
  Perturbations," ICLR 2019), with a manual NumPy/PIL fallback.
