"""Phase 8.1: multi-seed training + evaluation campaign driver.

Trains all four MobileNetV3-Small variants across multiple seeds on the
leak-free train/val/test split from Phase 8.0, then runs the full evaluation
pipeline (faithfulness, robustness, sanity) per (config, seed) cell. Each
stage is a `subprocess` call to the existing `scripts/*.py` entry points --
subprocess isolation keeps memory from one cell's run from leaking into the
next, which matters over a 20-cell, hours-scale campaign.

PAIRED-DESIGN INVARIANT (read before touching EVAL_SEED or the command
builders): faithfulness and robustness compare architectures/seeds against a
SHARED set of test images, so every (config, seed) cell must evaluate the
IDENTICAL image indices. Image selection (`select_indices` in
src/utils/script_helpers.py) is a deterministic function of the dataset,
`--num-images`, `--seed`, and `--stratified` -- nothing else. This driver
therefore ALWAYS passes the fixed `EVAL_SEED` (not the training seed `seed`)
as `--seed` to faithfulness_eval.py, robustness_eval.py, and sanity_check.py.
Never wire `cell.seed` into those commands' `--seed`: doing so would let the
training seed leak into eval-image selection and silently break the paired
design across cells.

RESUME SAFETY: --skip-existing (default on) makes every stage idempotent --
each stage checks for its own expected output artifact and skips (printing
SKIP) if it's already present and non-empty. Re-running this driver after a
Colab disconnect therefore resumes from wherever it stopped rather than
retraining/re-evaluating everything.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIGS = [
    "configs/vanilla_scratch.yaml",
    "configs/no_se_scratch.yaml",
    "configs/small_kernel_scratch.yaml",
    "configs/vanilla_finetune.yaml",
]
STAGE_ORDER = ["train", "faithfulness", "robustness", "sanity"]

# Fixed eval-sampling seed for every eval stage, independent of the training
# seed -- see the paired-design note in the module docstring. NEVER pass
# `cell.seed` here.
EVAL_SEED = 42

# Rough, hardware-agnostic per-stage time estimates for the plan preview only
# (a Colab T4-class GPU ballpark) -- not a guarantee, purely for planning.
ROUGH_SECONDS_PER_TRAIN_EPOCH = 90.0
ROUGH_SECONDS_PER_FAITHFULNESS_IMAGE = 1.2
ROUGH_SECONDS_PER_ROBUSTNESS_IMAGE = 4.0
ROUGH_SECONDS_SANITY_FLAT = 60.0


@dataclass
class Cell:
    config_path: Path
    experiment: str
    seed: int
    cell_dir: Path


def train_artifact_ok(cell_dir: Path) -> bool:
    ckpt = cell_dir / "checkpoints" / "best.pth"
    metrics = cell_dir / "metrics.json"
    return ckpt.exists() and ckpt.stat().st_size > 0 and metrics.exists() and metrics.stat().st_size > 0


def faithfulness_artifact_ok(cell_dir: Path) -> bool:
    p = cell_dir / "faithfulness" / "faithfulness_metrics.json"
    return p.exists() and p.stat().st_size > 0


def robustness_artifact_ok(cell_dir: Path) -> bool:
    p = cell_dir / "robustness" / "robustness_metrics.json"
    return p.exists() and p.stat().st_size > 0


def sanity_artifact_ok(cell_dir: Path) -> bool:
    p = cell_dir / "sanity" / "sanity_metrics.json"
    return p.exists() and p.stat().st_size > 0


STAGE_ARTIFACT_CHECK = {
    "train": train_artifact_ok,
    "faithfulness": faithfulness_artifact_ok,
    "robustness": robustness_artifact_ok,
    "sanity": sanity_artifact_ok,
}


def estimate_seconds(stage: str, cfg, args: argparse.Namespace) -> float:
    if stage == "train":
        return cfg.train.epochs * ROUGH_SECONDS_PER_TRAIN_EPOCH
    if stage == "faithfulness":
        return args.num_images_faithfulness * ROUGH_SECONDS_PER_FAITHFULNESS_IMAGE + 30.0
    if stage == "robustness":
        return args.num_images_robustness * ROUGH_SECONDS_PER_ROBUSTNESS_IMAGE + 30.0
    if stage == "sanity":
        return ROUGH_SECONDS_SANITY_FLAT
    raise ValueError(f"Unknown stage '{stage}'")


def build_cmd(stage: str, cell: Cell, args: argparse.Namespace) -> List[str]:
    python = sys.executable
    checkpoint = cell.cell_dir / "checkpoints" / "best.pth"

    if stage == "train":
        cmd = [
            python, "scripts/train.py",
            "--config", str(cell.config_path),
            "--seed", str(cell.seed),
            "--device", args.device,
            "--output-root", str(args.output_root),
        ]
    elif stage == "faithfulness":
        cmd = [
            python, "scripts/faithfulness_eval.py",
            "--checkpoints", str(checkpoint),
            "--num-images", str(args.num_images_faithfulness),
            "--seed", str(EVAL_SEED),
            "--output-dir", str(cell.cell_dir / "faithfulness"),
            "--device", args.device,
        ]
    elif stage == "robustness":
        cmd = [
            python, "scripts/robustness_eval.py",
            "--checkpoints", str(checkpoint),
            "--num-images", str(args.num_images_robustness),
            "--seed", str(EVAL_SEED),
            "--output-dir", str(cell.cell_dir / "robustness"),
            "--device", args.device,
        ]
    elif stage == "sanity":
        cmd = [
            python, "scripts/sanity_check.py",
            "--checkpoint", str(checkpoint),
            "--seed", str(EVAL_SEED),
            "--output-dir", str(cell.cell_dir / "sanity"),
            "--device", args.device,
        ]
    else:
        raise ValueError(f"Unknown stage '{stage}'")

    if args.no_download:
        cmd.append("--no-download")
    return cmd


def build_cells(configs: List[str], seeds: List[int], output_root: str) -> List[Cell]:
    cells = []
    for config in configs:
        config_path = Path(config)
        cfg = load_config(config_path)
        for seed in seeds:
            cell_dir = Path(output_root) / cfg.experiment / f"seed{seed}"
            cells.append(Cell(config_path=config_path, experiment=cfg.experiment, seed=seed, cell_dir=cell_dir))
    return cells


def print_plan(cells: List[Cell], stages: List[str], args: argparse.Namespace) -> Dict[str, tuple]:
    """Print the (config x seed) x stage matrix, existing-vs-remaining counts per
    stage, and a rough sequential-time estimate. Returns {stage: (existing, remaining)}."""
    cfg_cache: Dict[str, object] = {}

    def cfg_for(cell: Cell):
        key = str(cell.config_path)
        if key not in cfg_cache:
            cfg_cache[key] = load_config(cell.config_path)
        return cfg_cache[key]

    print("\n=== CAMPAIGN PLAN ===")
    header = f"{'experiment':<24}{'seed':>6}   " + "  ".join(f"{s:<12}" for s in stages)
    print(header)
    print("-" * len(header))
    for cell in cells:
        row = "  ".join(
            f"{'exists' if STAGE_ARTIFACT_CHECK[stage](cell.cell_dir) else 'pending':<12}" for stage in stages
        )
        print(f"{cell.experiment:<24}{cell.seed:>6}   {row}")

    counts: Dict[str, tuple] = {}
    total_seconds = 0.0
    total_remaining_seconds = 0.0
    print("\nPer-stage cell counts:")
    for stage in stages:
        existing = 0
        remaining = 0
        for cell in cells:
            est = estimate_seconds(stage, cfg_for(cell), args)
            total_seconds += est
            if STAGE_ARTIFACT_CHECK[stage](cell.cell_dir):
                existing += 1
            else:
                remaining += 1
                total_remaining_seconds += est
        counts[stage] = (existing, remaining)
        print(f"  {stage:<14} existing={existing:<4} remaining={remaining:<4}")

    print(
        f"\nRough estimated remaining sequential time: "
        f"~{total_remaining_seconds / 3600:.1f} h ({total_remaining_seconds / 60:.0f} min)"
    )
    print(f"Rough estimated total time if nothing were skipped: ~{total_seconds / 3600:.1f} h")
    print(
        "(These are hardware-agnostic ballpark estimates for planning only -- "
        "not a guarantee of actual runtime.)"
    )
    return counts


def run_stage(stage: str, cell: Cell, args: argparse.Namespace) -> str:
    """Run one (cell, stage). Returns one of "skip", "blocked", "done", "failed"."""
    cell.cell_dir.mkdir(parents=True, exist_ok=True)
    label = f"[{cell.experiment} seed={cell.seed} stage={stage}]"

    if args.skip_existing and STAGE_ARTIFACT_CHECK[stage](cell.cell_dir):
        print(f"{label} SKIP (exists)")
        return "skip"

    if stage != "train":
        checkpoint = cell.cell_dir / "checkpoints" / "best.pth"
        if not checkpoint.exists():
            print(f"{label} SKIP (checkpoint missing -- run the 'train' stage first)")
            return "blocked"

    print(f"{label} STARTING")
    cmd = build_cmd(stage, cell, args)
    log_path = cell.cell_dir / f"{stage}.log"
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))

    if result.returncode != 0:
        print(f"{label} FAILED (exit {result.returncode}; see {log_path})")
        return "failed"
    if not STAGE_ARTIFACT_CHECK[stage](cell.cell_dir):
        print(f"{label} FAILED (exit 0 but expected artifact is missing; see {log_path})")
        return "failed"

    print(f"{label} DONE (log: {log_path})")
    return "done"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8.1 multi-seed training + evaluation campaign driver.")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--num-images-faithfulness", type=int, default=500)
    parser.add_argument("--num-images-robustness", type=int, default=200)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-root", default="runs", help="Root directory for run outputs.")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a stage if its expected artifact already exists (default: on) -- resume safety.",
    )
    parser.add_argument(
        "--stages", nargs="+", choices=STAGE_ORDER, default=STAGE_ORDER,
        help="Subset of stages to run (order is always train,faithfulness,robustness,sanity).",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Pass --no-download through to every stage (synthetic data; smoke testing only).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit without running anything.")
    args = parser.parse_args()
    args.stages = [s for s in STAGE_ORDER if s in args.stages]
    return args


def main() -> Dict[str, int]:
    args = parse_args()
    cells = build_cells(args.configs, args.seeds, args.output_root)

    print_plan(cells, args.stages, args)

    tally = {"skip": 0, "blocked": 0, "done": 0, "failed": 0}
    if args.dry_run:
        print("\n--dry-run: stopping before running anything.")
        return tally

    print("\n=== RUNNING ===")
    failures = []
    for cell in cells:
        for stage in args.stages:
            outcome = run_stage(stage, cell, args)
            tally[outcome] += 1
            if outcome == "failed":
                failures.append((cell.experiment, cell.seed, stage))

    print("\n=== CAMPAIGN COMPLETE ===")
    print(f"done={tally['done']} skip={tally['skip']} blocked={tally['blocked']} failed={tally['failed']}")
    if failures:
        print("Failed stages:")
        for exp, seed, stage in failures:
            print(f"  {exp} seed={seed} stage={stage}")
    return tally


if __name__ == "__main__":
    main()
