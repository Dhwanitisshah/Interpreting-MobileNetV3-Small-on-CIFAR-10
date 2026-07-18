# Changelog

All notable changes to this project are documented here. This project follows
its own phase-based development log rather than semantic versioning, since it
is an active research codebase rather than a versioned library.

## Phase 7.5 — 2026-07-19

- **Fixed a real evaluation-reproducibility bug**: `scripts/robustness_eval.py`
  seeded only the local image-selection RNG, leaving NumPy's global RNG
  unseeded — and `imagecorruptions` draws its noise realizations from that
  global state, so reruns with an identical `--seed` and checkpoints could
  silently produce different drift numbers (measured pre-fix variation: ~0.003
  on mean drift values of ~0.22). Added `set_seed(args.seed)` at the top of
  `main()` in `robustness_eval.py`, `faithfulness_eval.py`, `sanity_check.py`,
  and `concentration_diagnostic.py`.
- Added a reproducibility assertion to `scripts/smoke_test_robustness.py`:
  two in-process reruns of `evaluate_robustness` with the seed reset between
  them must produce bitwise-identical drift fields (not `np.allclose`).
- Verified on real data: two independent full robustness runs
  (`runs/robustness_fixed_seeded/`, `runs/robustness_reprocheck/`; same
  checkpoints, `--num-images 200`, `--seed 42`) were diffed programmatically —
  all 57,600 drift-field values across 14,400 records were bitwise-identical.
- **Updated README.md's canonical robustness numbers** to source from
  `runs/robustness_fixed_seeded/` instead of the pre-fix `runs/robustness/`.
  The qualitative finding survives (drift still scales with architecture
  beyond accuracy loss) but the reported magnitudes changed, and the wording
  was corrected to not overstate: the pre-training (`vanilla_finetune`) drift
  increase is directional (elevated on `motion_blur`/`jpeg_compression`/
  `defocus_blur`, negligible on `brightness`, reversed on `contrast`), not
  general; the SE-ablation drift equivalence holds on 3 of 4 metrics at SESOI
  0.3 Cohen's d, with the 4th (centroid shift) inconclusive rather than
  equivalent. `runs/robustness/` and `runs/robustness_fixed/` (both gitignored,
  pre-fix) were left untouched as a record of the earlier runs.

## [Unreleased] — Repository cleanup

- De-duplicated device resolution, checkpoint loading, synthetic-dataset
  fallback, stratified index sampling, and the shared plotting palette
  (previously copy-pasted across `scripts/*.py`) into `src/utils/script_helpers.py`.
- Moved the TOST equivalence test (`tost_paired`) and its default SESOI
  constant into `src/metrics/equivalence.py`, replacing a fragile
  script-importing-script pattern in `report_robustness.py` and
  `concentration_diagnostic.py`.
- Moved `DRIFT_METRICS`, `HIGHER_IS_MORE_DRIFT`, and `drift_score` into
  `src/robustness/drift.py` alongside the existing `TOP_K_FRACTION` constant.
- Added a professional `README.md`, `LICENSE` (MIT), `CONTRIBUTING.md`,
  `weights/README.md`, and an improved `.gitignore`.
- Verified the refactor is behavior-preserving: all smoke tests pass, and
  `report_faithfulness.py` / `report_robustness.py` / `concentration_diagnostic.py`
  produce byte-identical output on the existing `runs/` artifacts before and
  after the refactor.

## Phase 7.4 — 2026-07-18

- Accuracy-floor sensitivity analysis: drops (corruption, severity) cells
  where any model's accuracy falls to chance level before recomputing drift
  statistics, to confirm findings aren't artifacts of near-random predictions.

## Phase 7.3 — 2026-07-18

- Fixed `explanation_drift` to compare CAMs against a fixed target class (the
  clean prediction), rather than the corrupted model's own (possibly
  different) prediction, and reran Phase 7.

## Phase 7.2 — 2026-07-18

- Per-corruption breakdown of explanation drift.

## Phase 7.1 — 2026-07-18

- CAM concentration diagnostic and drift equivalence (TOST) testing.

## Phase 7 — 2026-07-18

- Explanation robustness under distribution shift: Grad-CAM drift across six
  ImageNet-C-style corruptions and three severities, per architectural
  variant.

## Phase 6.3 — 2026-07-18

- p0-confound diagnostic ruling out confidence-normalization artifacts as the
  source of cross-model faithfulness ranking differences.

## Phase 6.2 — 2026-07-18

- Faithfulness reporting layer with TOST equivalence testing.

## Phase 6.1 — 2026-07-18

- Fixed checkpoint loading and added confidence-normalized faithfulness
  metrics.

## Phase 6 — 2026-07-18

- Quantitative Grad-CAM faithfulness metrics (deletion/insertion AUC, ROAD
  gap) with paired statistical significance testing across model variants.

## Phase 5 — 2026-07-18

- Cross-variant Grad-CAM comparison (`vanilla`, `no_se`, `small_kernel`).

## Phase 4 — 2026-07-16

- Grad-CAM sanity checks: cascading (top-down) parameter-randomization test
  (Adebayo et al., 2018) with Spearman and SSIM similarity metrics.

## Phase 3 — 2026-07-15

- From-scratch Grad-CAM module (Selvaraju et al., 2017), independent of any
  third-party CAM library.

## Phase 2 — 2026-07-15

- Training and evaluation harness: SGD with cosine annealing, checkpointing,
  per-class accuracy, confusion matrix, and correct/incorrect prediction
  index artifacts.

## Phase 1 — 2026-07-15

- Initial scaffold: deterministic seeding, YAML configs with dot-access
  loading, the CIFAR-10 data pipeline, and the three MobileNetV3-Small
  variants (`vanilla`, `no_se`, `small_kernel`).
