"""Smoke test for scripts/run_campaign.py (Phase 8.1). No download, no real
training: points the driver at a throwaway tiny synthetic config (epochs=1,
--no-download, a handful of eval images) and runs 2 seeds x 1 config x the
default 3 attribution methods (gradcam, gradcampp, random) through all four
stages into a temp output root, exercising the real subprocess-based driver
end to end.

Asserts:
  - every expected artifact path (runs/<exp>/seed<N>/<method>/<stage>/) is
    created for both seeds and every method,
  - a second invocation with --skip-existing (the default) skips every job
    and does no work,
  - faithfulness/robustness eval image indices are identical across the two
    seeds AND across the three methods (the paired-design invariant the
    driver is built to protect), and each output records its own method, and
  - the plan-preview counts (existing vs. remaining) are correct both before
    and after the run: 2 train jobs + 2 seeds x 3 methods x 3 eval stages.
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
METHODS = ["gradcam", "gradcampp", "random"]
EVAL_STAGES = ["faithfulness", "robustness", "sanity"]


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


def load_metrics(output_root: Path, seed: int, method: str, stage: str, metrics_file: str) -> dict:
    path = output_root / "smoke_campaign_exp" / f"seed{seed}" / method / stage / metrics_file
    with open(path) as f:
        data = json.load(f)
    (only_model,) = data.keys()
    return data[only_model]


def load_eval_indices(output_root: Path, seed: int, method: str, stage: str, metrics_file: str) -> list:
    return [r["index"] for r in load_metrics(output_root, seed, method, stage, metrics_file)["records"]]


def main() -> bool:
    results = []

    # --- --configs accepts a bare name, a bare filename, and a full path ---
    bare = run_campaign.resolve_config_path("vanilla_scratch")
    with_ext = run_campaign.resolve_config_path("vanilla_scratch.yaml")
    full_path = run_campaign.resolve_config_path("configs/vanilla_scratch.yaml")
    results.append(("resolve_config_path: bare name and full path agree", bare == full_path))
    results.append(("resolve_config_path: bare filename and full path agree", with_ext == full_path))

    cells_bare = run_campaign.build_cells(["vanilla_scratch"], [0], "runs")
    cells_path = run_campaign.build_cells(["configs/vanilla_scratch.yaml"], [0], "runs")
    results.append(
        ("build_cells: bare name and full path resolve to the same cell",
         cells_bare[0].config_path == cells_path[0].config_path
         and cells_bare[0].experiment == cells_path[0].experiment
         and cells_bare[0].cell_dir == cells_path[0].cell_dir)
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        config_path = tmp_path / "tiny_config.yaml"
        config_path.write_text(TINY_CONFIG_YAML)
        output_root = tmp_path / "runs"

        argv = make_argv(config_path, output_root)

        # --- BEFORE: plan preview shows every cell pending ---
        args_before = with_argv(argv, run_campaign.parse_args)
        cells = run_campaign.build_cells(args_before.configs, args_before.seeds, args_before.output_root)
        results.append(("default --methods is gradcam, gradcampp, random", args_before.methods == METHODS))
        results.append(("plan builds 2 cells (1 config x 2 seeds)", len(cells) == 2))

        with redirect_stdout(io.StringIO()):
            counts_before = run_campaign.print_plan(cells, args_before.stages, args_before)
        results.append(
            ("plan preview: train is 2 jobs, each eval stage 6 jobs (2 seeds x 3 methods), all pending",
             counts_before == {"train": (0, 2), "faithfulness": (0, 6), "robustness": (0, 6), "sanity": (0, 6)})
        )

        # --- FIRST RUN: full campaign, 2 seeds x 1 config x 4 stages ---
        tally = with_argv(argv, run_campaign.main)
        results.append(("first run: nothing blocked or failed", tally["blocked"] == 0 and tally["failed"] == 0))
        results.append(("first run: nothing was skipped", tally["skip"] == 0))
        results.append(("first run: 2 train + 2 seeds x 3 methods x 3 eval stages = 20 jobs ran", tally["done"] == 20))

        for seed in SEEDS:
            cell_dir = output_root / "smoke_campaign_exp" / f"seed{seed}"
            results.append((f"seed{seed}: train artifact exists (one per cell, method-independent)", run_campaign.train_artifact_ok(cell_dir)))
            for method in METHODS:
                for stage, check in (
                    ("faithfulness", run_campaign.faithfulness_artifact_ok),
                    ("robustness", run_campaign.robustness_artifact_ok),
                    ("sanity", run_campaign.sanity_artifact_ok),
                ):
                    results.append((f"seed{seed}/{method}/{stage}: artifact exists", check(cell_dir, method)))
            results.append((f"seed{seed}: no un-nested (method-less) eval output dirs",
                            not any((cell_dir / stage).exists() for stage in EVAL_STAGES)))

        # --- AFTER: plan preview shows every job existing ---
        with redirect_stdout(io.StringIO()):
            counts_after = run_campaign.print_plan(cells, args_before.stages, args_before)
        results.append(
            ("plan preview: every job exists after running",
             counts_after == {"train": (2, 0), "faithfulness": (6, 0), "robustness": (6, 0), "sanity": (6, 0)})
        )

        # --- SECOND RUN: --skip-existing (default) must skip everything, no work done ---
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            tally2 = with_argv(argv, run_campaign.main)
        results.append(
            ("second run: every job skipped (resume safety)", tally2["skip"] == 20 and tally2["done"] == 0)
        )
        results.append(
            ("second run's output has no STARTING/DONE lines (no work was done)",
             "STARTING" not in buf2.getvalue() and "DONE" not in buf2.getvalue())
        )

        # --- Paired-design check: eval image indices identical across seeds AND methods ---
        for stage, metrics_file in (("faithfulness", "faithfulness_metrics.json"), ("robustness", "robustness_metrics.json")):
            idx = {
                (seed, method): sorted(set(load_eval_indices(output_root, seed, method, stage, metrics_file)))
                for seed in SEEDS
                for method in METHODS
            }
            results.append((f"paired design: {stage} eval indices identical across seeds",
                            all(idx[(SEEDS[0], m)] == idx[(SEEDS[1], m)] for m in METHODS)))
            results.append((f"paired design: {stage} eval indices identical across methods",
                            all(idx[(sd, METHODS[0])] == idx[(sd, m)] for sd in SEEDS for m in METHODS)))

        # --- each output records the method it was produced with ---
        for stage, metrics_file in (("faithfulness", "faithfulness_metrics.json"), ("robustness", "robustness_metrics.json")):
            results.append((f"{stage}: every output records its own method", all(
                load_metrics(output_root, seed, method, stage, metrics_file)["method"] == method
                for seed in SEEDS for method in METHODS)))

        # --- the random method really ran a different explainer than gradcam ---
        gc_f = load_metrics(output_root, SEEDS[0], "gradcam", "faithfulness", "faithfulness_metrics.json")["aggregate"]
        rnd_f = load_metrics(output_root, SEEDS[0], "random", "faithfulness", "faithfulness_metrics.json")["aggregate"]
        results.append(("random method's faithfulness differs from gradcam's (different explainers ran)",
                        gc_f["deletion_auc_mean"] != rnd_f["deletion_auc_mean"]))

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
