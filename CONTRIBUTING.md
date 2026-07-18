# Contributing

This is a research codebase accompanying an in-progress paper on
interpretability and distribution-shift robustness in MobileNetV3 variants.
Contributions are welcome, particularly:

- Bug reports and fixes in the Grad-CAM, faithfulness, or robustness metric
  implementations.
- Additional architectural variants or datasets, added as new
  `configs/*.yaml` files without touching existing experiment configs.
- Documentation and reproducibility improvements.

## Development setup

```powershell
git clone <this-repo>
cd RESEARCH_PAPER
pip install -r requirements.txt
```

## Before submitting a change

1. Run the smoke tests (no CIFAR-10 download required):

   ```powershell
   python scripts/smoke_test.py
   python scripts/smoke_test_train.py
   python scripts/smoke_test_gradcam.py
   python scripts/smoke_test_sanity.py
   python scripts/smoke_test_compare.py
   python scripts/smoke_test_faithfulness.py
   python scripts/smoke_test_robustness.py
   ```

2. If you changed a metric or statistical test, verify it against an existing
   `runs/` artifact where possible (e.g. rerun `scripts/report_faithfulness.py`
   and confirm the numbers you expect to be unaffected are unaffected).

3. Keep changes scoped: this project prioritizes methodological correctness
   and reproducibility over new features. If you're proposing a new
   experiment axis, open an issue first to discuss scope.

## Code style

- Prefer `pathlib.Path` over string path manipulation.
- Add type hints on new public functions.
- Add a docstring where the *why*, not the *what*, needs explaining — the
  existing `src/metrics/faithfulness.py` and `src/robustness/drift.py` are
  good examples of the level of detail expected for anything touching the
  statistical methodology.
- Avoid duplicating logic already in `src/utils/script_helpers.py`,
  `src/metrics/`, or `src/robustness/` — these hold the helpers shared across
  `scripts/*.py` entry points.
