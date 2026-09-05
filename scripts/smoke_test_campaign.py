"""Smoke test for scripts/run_campaign.py (Phase 8.1). No download, no real
training: points the driver at a throwaway tiny synthetic config (epochs=1,
--no-download, a handful of eval images) and runs 2 seeds x 1 config through
all four stages into a temp output root, exercising the real subprocess-based
driver end to end.

Asserts:
  - every expected artifact path is created for both seeds,
  - a second invocation with --skip-existing (the default) skips every cell
    and does no work,
  - faithfulness/robustness eval image indices are identical across the two
    seeds (the paired-design invariant the driver is built to protect), and
  - the plan-preview counts (existing vs. remaining) are correct both before
    and after the run.
"""

import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/, to import run_campaign

import run_campaign

TINY_CONFIG_YAML = """
experiment: smoke_campaign_exp
seed: 42
model:
  variant: vanilla
  pretrained: false
  num_classes: 10
data:
  root: ./data
  train_batch_size: 8
  test_batch_size: 8
  num_workers: 0
train:
  epochs: 1
  optimizer: sgd
  lr: 0.01
  momentum: 0.9
  weight_decay: 0.0005
  scheduler: cosine
"""

SEEDS = [100, 200]


def make_argv(config_path: Path, output_root: Path) -> list:
    return [
        "--configs", str(config_path),
        "--seeds", *[str(s) for s in SEEDS],
        "--num-images-faithfulness", "4",
        "--num-images-robustness", "4",
        "--device", "cpu",
        "--output-root", str(output_root),
        "--no-download",
    ]


def with_argv(argv: list, fn):
    old_argv = sys.argv
    sys.argv = ["run_campaign.py"] + argv
    try:
        return fn()
    finally:
        sys.argv = old_argv


def load_eval_indices(output_root: Path, seed: int, stage: str, metrics_file: str) -> list:
    path = output_root / "smoke_campaign_exp" / f"seed{seed}" / stage / metrics_file
    with open(path) as f:
        data = json.load(f)
    (only_model,) = data.keys()
    return [r["index"] for r in data[only_model]["records"]]


def main() -> bool:
    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        config_path = tmp_path / "tiny_config.yaml"
        config_path.write_text(TINY_CONFIG_YAML)
        output_root = tmp_path / "runs"

        argv = make_argv(config_path, output_root)

        # --- BEFORE: plan preview shows every cell pending ---
        args_before = with_argv(argv, run_campaign.parse_args)
        cells = run_campaign.build_cells(args_before.configs, args_before.seeds, args_before.output_root)
        results.append(("plan builds 2 cells (1 config x 2 seeds)", len(cells) == 2))

        with redirect_stdout(io.StringIO()):
            counts_before = run_campaign.print_plan(cells, args_before.stages, args_before)
        results.append(
            ("plan preview: all cells pending before running anything",
             all(existing == 0 and remaining == 2 for existing, remaining in counts_before.values()))
        )

        # --- FIRST RUN: full campaign, 2 seeds x 1 config x 4 stages ---
        tally = with_argv(argv, run_campaign.main)
        results.append(("first run: nothing blocked or failed", tally["blocked"] == 0 and tally["failed"] == 0))
        results.append(("first run: nothing was skipped", tally["skip"] == 0))
        results.append(("first run: all 2 seeds x 4 stages = 8 cells ran", tally["done"] == 8))

        for seed in SEEDS:
            cell_dir = output_root / "smoke_campaign_exp" / f"seed{seed}"
            results.append((f"seed{seed}: train artifact exists", run_campaign.train_artifact_ok(cell_dir)))
            results.append((f"seed{seed}: faithfulness artifact exists", run_campaign.faithfulness_artifact_ok(cell_dir)))
            results.append((f"seed{seed}: robustness artifact exists", run_campaign.robustness_artifact_ok(cell_dir)))
            results.append((f"seed{seed}: sanity artifact exists", run_campaign.sanity_artifact_ok(cell_dir)))

        # --- AFTER: plan preview shows every cell existing ---
        with redirect_stdout(io.StringIO()):
            counts_after = run_campaign.print_plan(cells, args_before.stages, args_before)
        results.append(
            ("plan preview: all cells exist after running",
             all(existing == 2 and remaining == 0 for existing, remaining in counts_after.values()))
        )

        # --- SECOND RUN: --skip-existing (default) must skip everything, no work done ---
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            tally2 = with_argv(argv, run_campaign.main)
        results.append(
            ("second run: every cell skipped (resume safety)", tally2["skip"] == 8 and tally2["done"] == 0)
        )
        results.append(
            ("second run's output has no STARTING/DONE lines (no work was done)",
             "STARTING" not in buf2.getvalue() and "DONE" not in buf2.getvalue())
        )

        # --- Paired-design check: eval image indices identical across seeds ---
        faith_0 = load_eval_indices(output_root, SEEDS[0], "faithfulness", "faithfulness_metrics.json")
        faith_1 = load_eval_indices(output_root, SEEDS[1], "faithfulness", "faithfulness_metrics.json")
        results.append(
            ("paired design: faithfulness eval indices identical across seeds", faith_0 == faith_1)
        )

        rob_0 = load_eval_indices(output_root, SEEDS[0], "robustness", "robustness_metrics.json")
        rob_1 = load_eval_indices(output_root, SEEDS[1], "robustness", "robustness_metrics.json")
        results.append(
            ("paired design: robustness eval indices identical across seeds",
             sorted(set(rob_0)) == sorted(set(rob_1)))
        )

    print("\n=== SMOKE TEST RESULTS ===")
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"[{status}] {name}")

    print("\n" + ("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED"))
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
