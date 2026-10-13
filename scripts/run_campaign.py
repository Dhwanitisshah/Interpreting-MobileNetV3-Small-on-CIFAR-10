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

ATTRIBUTION METHODS (Phase 8.2): each (config, seed) cell runs the eval stages
(faithfulness, robustness, sanity) once per method in --methods (default:
gradcam gradcampp random -- the real method, a second real method, and the
uniform-noise null floor), writing to
runs/<exp>/seed<N>/<method>/<stage>/. Training is method-independent and runs
once per cell, into runs/<exp>/seed<N>/. EVAL_SEED stays fixed for every
method too, so all methods see the identical images (and `random` is the same
noise map for the same image in every cell).

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
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explain import METHODS
from src.utils import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIGS = [
    "vanilla_scratch",
    "no_se_scratch",
    "small_kernel_scratch",
    "vanilla_finetune",
]
STAGE_ORDER = ["train", "faithfulness", "robustness", "sanity"]
EVAL_STAGES = [s for s in STAGE_ORDER if s != "train"]
DEFAULT_METHODS = list(METHODS)

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


def eval_dir(cell_dir: Path, method: str, stage: str) -> Path:
    """Where an eval stage's outputs live: runs/<exp>/seed<N>/<method>/<stage>/."""
    return cell_dir / method / stage


def train_artifact_ok(cell_dir: Path, method: Optional[str] = None) -> bool:
    ckpt = cell_dir / "checkpoints" / "best.pth"
    metrics = cell_dir / "metrics.json"
    return ckpt.exists() and ckpt.stat().st_size > 0 and metrics.exists() and metrics.stat().st_size > 0


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def faithfulness_artifact_ok(cell_dir: Path, method: str) -> bool:
    return _nonempty(eval_dir(cell_dir, method, "faithfulness") / "faithfulness_metrics.json")


def robustness_artifact_ok(cell_dir: Path, method: str) -> bool:
    return _nonempty(eval_dir(cell_dir, method, "robustness") / "robustness_metrics.json")


def sanity_artifact_ok(cell_dir: Path, method: str) -> bool:
    return _nonempty(eval_dir(cell_dir, method, "sanity") / "sanity_metrics.json")


STAGE_ARTIFACT_CHECK = {
    "train": train_artifact_ok,
    "faithfulness": faithfulness_artifact_ok,
    "robustness": robustness_artifact_ok,
    "sanity": sanity_artifact_ok,
}


def stage_methods(stage: str, methods: List[str]) -> List[Optional[str]]:
    """Methods a stage runs for: training is method-independent (one job per cell,
    method=None); each eval stage runs once per method."""
    return [None] if stage == "train" else list(methods)


def stage_done(stage: str, cell: "Cell", method: Optional[str]) -> bool:
    return STAGE_ARTIFACT_CHECK[stage](cell.cell_dir, method)


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


def build_cmd(stage: str, cell: Cell, args: argparse.Namespace, method: Optional[str] = None) -> List[str]:
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
            "--method", method,
            "--output-dir", str(eval_dir(cell.cell_dir, method, "faithfulness")),
            "--device", args.device,
        ]
    elif stage == "robustness":
        cmd = [
            python, "scripts/robustness_eval.py",
            "--checkpoints", str(checkpoint),
            "--num-images", str(args.num_images_robustness),
            "--seed", str(EVAL_SEED),
            "--method", method,
            "--output-dir", str(eval_dir(cell.cell_dir, method, "robustness")),
            "--device", args.device,
        ]
    elif stage == "sanity":
        cmd = [
            python, "scripts/sanity_check.py",
            "--checkpoint", str(checkpoint),
            "--seed", str(EVAL_SEED),
            "--method", method,
            "--output-dir", str(eval_dir(cell.cell_dir, method, "sanity")),
            "--device", args.device,
        ]
    else:
        raise ValueError(f"Unknown stage '{stage}'")

    if args.no_download:
        cmd.append("--no-download")
    return cmd


def resolve_config_path(value: str) -> Path:
    """Resolve a --configs entry to a real YAML file.

    Accepts, in order: the value as-is (a path, relative to the current
    working directory, or absolute -- e.g. "configs/vanilla_scratch.yaml" or
    "/abs/path/foo.yaml"); a bare name under configs/ ("vanilla_scratch.yaml");
    or a bare experiment name with the extension implied ("vanilla_scratch").
    The configs/ fallbacks are anchored to the repo root so bare names resolve
    the same way regardless of the driver's working directory.
    """
    candidates = [
        Path(value),
        REPO_ROOT / "configs" / value,
        REPO_ROOT / "configs" / f"{value}.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            # Resolve to an absolute path: stage subprocesses run with
            # cwd=REPO_ROOT (see run_stage), which would silently mis-resolve
            # a relative "as-is" match taken from the driver's own cwd.
            return candidate.resolve()
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not resolve --configs value '{value}'; tried: {tried}")


def build_cells(configs: List[str], seeds: List[int], output_root: str) -> List[Cell]:
    cells = []
    for config in configs:
        config_path = resolve_config_path(config)
        cfg = load_config(config_path)
        for seed in seeds:
            cell_dir = Path(output_root) / cfg.experiment / f"seed{seed}"
            cells.append(Cell(config_path=config_path, experiment=cfg.experiment, seed=seed, cell_dir=cell_dir))
    return cells


def print_plan(cells: List[Cell], stages: List[str], args: argparse.Namespace) -> Dict[str, tuple]:
    """Print the (config x seed x method) x stage matrix, existing-vs-remaining job
    counts per stage, and a rough sequential-time estimate. Training is one job
    per (config, seed); each eval stage is one job per (config, seed, method).
    Returns {stage: (existing, remaining)}."""
    cfg_cache: Dict[str, object] = {}

    def cfg_for(cell: Cell):
        key = str(cell.config_path)
        if key not in cfg_cache:
            cfg_cache[key] = load_config(cell.config_path)
        return cfg_cache[key]

    print("\n=== CAMPAIGN PLAN ===")
    print(f"methods: {', '.join(args.methods)}")
    header = f"{'experiment':<24}{'seed':>6}  {'method':<10}" + "  ".join(f"{s:<12}" for s in stages)
    print(header)
    print("-" * len(header))
    for cell in cells:
        for m_idx, method in enumerate(args.methods):
            row_cells = []
            for stage in stages:
                if stage == "train":
                    # one train job per cell: show it on the cell's first row only
                    text = ("exists" if stage_done(stage, cell, None) else "pending") if m_idx == 0 else "-"
                else:
                    text = "exists" if stage_done(stage, cell, method) else "pending"
                row_cells.append(f"{text:<12}")
            print(f"{cell.experiment:<24}{cell.seed:>6}  {method:<10}" + "  ".join(row_cells))

    counts: Dict[str, tuple] = {}
    total_seconds = 0.0
    total_remaining_seconds = 0.0
    print("\nPer-stage job counts (train: per config x seed; eval stages: per config x seed x method):")
    for stage in stages:
        existing = 0
        remaining = 0
        for cell in cells:
            for method in stage_methods(stage, args.methods):
                est = estimate_seconds(stage, cfg_for(cell), args)
                total_seconds += est
                if stage_done(stage, cell, method):
                    existing += 1
                else:
                    remaining += 1
                    total_remaining_seconds += est
        counts[stage] = (existing, remaining)
        print(f"  {stage:<14} existing={existing:<4} remaining={remaining:<4}")
    total_jobs = sum(e + r for e, r in counts.values())
    total_remaining = sum(r for _, r in counts.values())
    print(f"  {'TOTAL':<14} jobs={total_jobs:<4} remaining={total_remaining:<4}")

    print(
        f"\nRough estimated remaining sequential time: "
        f"~{total_remaining_seconds / 3600:.1f} h ({total_remaining_seconds / 60:.0f} min)"
    )
    print(f"Rough estimated total time if nothing were skipped: ~{total_seconds / 3600:.1f} h")
    print(
        "(These are hardware-agnostic ballpark estimates for planning only -- "
        "not a guarantee of actual runtime; every method is costed like Grad-CAM, so the "
        "cheaper random baseline makes this a mild overestimate.)"
    )
    return counts


def run_stage(stage: str, cell: Cell, args: argparse.Namespace, method: Optional[str] = None) -> str:
    """Run one (cell, stage, method) job. `method` is None for the train stage.
    Returns one of "skip", "blocked", "done", "failed"."""
    cell.cell_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{stage}" if method is None else f"{stage} method={method}"
    label = f"[{cell.experiment} seed={cell.seed} stage={tag}]"

    if args.skip_existing and stage_done(stage, cell, method):
        print(f"{label} SKIP (exists)")
        return "skip"

    if stage != "train":
        checkpoint = cell.cell_dir / "checkpoints" / "best.pth"
        if not checkpoint.exists():
            print(f"{label} SKIP (checkpoint missing -- run the 'train' stage first)")
            return "blocked"

    print(f"{label} STARTING")
    cmd = build_cmd(stage, cell, args, method)
    log_dir = cell.cell_dir if method is None else cell.cell_dir / method
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage}.log"
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))

    if result.returncode != 0:
        print(f"{label} FAILED (exit {result.returncode}; see {log_path})")
        return "failed"
    if not stage_done(stage, cell, method):
        print(f"{label} FAILED (exit 0 but expected artifact is missing; see {log_path})")
        return "failed"

    print(f"{label} DONE (log: {log_path})")
    return "done"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8.1 multi-seed training + evaluation campaign driver.")
    parser.add_argument(
        "--configs", nargs="+", default=DEFAULT_CONFIGS,
        help="Configs to run: bare experiment names (vanilla_scratch), bare filenames "
        "(vanilla_scratch.yaml), or paths (configs/vanilla_scratch.yaml) -- see "
        "resolve_config_path. Default: all four project configs, by bare name.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--methods", nargs="+", choices=list(METHODS), default=DEFAULT_METHODS,
        help="Attribution methods to evaluate in each eval stage (faithfulness/robustness/sanity): "
        "gradcam, gradcampp (Grad-CAM++), random (uniform-noise null baseline). Default: all three. "
        "Outputs go to <output-root>/<exp>/seed<N>/<method>/<stage>/.",
    )
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
        # Per cell: train once, then every eval stage for every method.
        jobs = [("train", None)] if "train" in args.stages else []
        for method in args.methods:
            jobs.extend((stage, method) for stage in EVAL_STAGES if stage in args.stages)
        for stage, method in jobs:
            outcome = run_stage(stage, cell, args, method)
            tally[outcome] += 1
            if outcome == "failed":
                failures.append((cell.experiment, cell.seed, stage if method is None else f"{stage}/{method}"))

    print("\n=== CAMPAIGN COMPLETE ===")
    print(f"done={tally['done']} skip={tally['skip']} blocked={tally['blocked']} failed={tally['failed']}")
    if failures:
        print("Failed stages:")
        for exp, seed, stage in failures:
            print(f"  {exp} seed={seed} stage={stage}")
    return tally


if __name__ == "__main__":
    main()
