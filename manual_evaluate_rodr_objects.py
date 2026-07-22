import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


STEP_RE = re.compile(r"step_(\d+)\.pth$")
SETTING_RE = re.compile(r".*_steps_(?P<steps>\d+)_(?P<resolution>\d+)_(?P<noise>[0-9.]+)$")


@dataclass(frozen=True)
class CheckpointPair:
    step: int
    baseline: Path
    candidate: Path
    candidate_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find matching baseline/RODR checkpoints, evaluate them on PUNet and PCNet, "
            "and write a CD/P2M comparison table."
        )
    )
    parser.add_argument("--save_root", default="experiments/rodr_compare_punet_full")
    parser.add_argument("--baseline_name", default="baseline_mse")
    parser.add_argument("--candidate_glob", default="rodr_w*")
    parser.add_argument("--step", type=int, default=None, help="Evaluate one shared checkpoint step.")
    parser.add_argument("--all_steps", action="store_true", help="Evaluate every shared checkpoint step.")
    parser.add_argument("--datasets", nargs="+", default=["PUNet", "PCNet"], choices=["PUNet", "PCNet"])
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--dataset_root", default="./data/objects")
    parser.add_argument("--data_path", default="./data/objects/examples")
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--diffusion_steps", type=int, default=5)
    parser.add_argument("--k", type=int, default=3, help="Patch oversampling factor passed to evaluate_objects.py.")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--save_intermediate", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run evaluation even if summary CSVs already exist.")
    return parser.parse_args()


def collect_step_checkpoints(ckpt_dir: Path) -> dict[int, Path]:
    ckpts: dict[int, Path] = {}
    for path in sorted(ckpt_dir.glob("step_*.pth")):
        match = STEP_RE.match(path.name)
        if match:
            ckpts[int(match.group(1))] = path
    return ckpts


def find_pairs(args: argparse.Namespace) -> list[CheckpointPair]:
    save_root = Path(args.save_root)
    baseline_dir = save_root / args.baseline_name
    baseline_ckpts = collect_step_checkpoints(baseline_dir)
    if not baseline_ckpts:
        raise FileNotFoundError(f"No baseline checkpoints found under {baseline_dir}")

    candidate_dirs = sorted(path for path in save_root.glob(args.candidate_glob) if path.is_dir())
    if not candidate_dirs:
        raise FileNotFoundError(f"No candidate dirs match {save_root / args.candidate_glob}")

    pairs: list[CheckpointPair] = []
    for candidate_dir in candidate_dirs:
        candidate_ckpts = collect_step_checkpoints(candidate_dir)
        common_steps = sorted(set(baseline_ckpts) & set(candidate_ckpts))
        if args.step is not None:
            common_steps = [args.step] if args.step in common_steps else []
        elif not args.all_steps and common_steps:
            common_steps = [common_steps[-1]]

        for step in common_steps:
            pairs.append(
                CheckpointPair(
                    step=step,
                    baseline=baseline_ckpts[step],
                    candidate=candidate_ckpts[step],
                    candidate_name=candidate_dir.name,
                )
            )

    if not pairs:
        raise FileNotFoundError("No matching baseline/candidate checkpoint steps found.")
    return pairs


def summary_files(output_dir: Path, dataset: str) -> list[Path]:
    return sorted(output_dir.rglob(f"Summary_{dataset}.csv"))


def has_completed_eval(output_dir: Path, dataset: str) -> bool:
    return bool(summary_files(output_dir, dataset))


def run_evaluate_objects(
    ckpt: Path,
    output_dir: Path,
    dataset: str,
    args: argparse.Namespace,
) -> None:
    if has_completed_eval(output_dir, dataset) and not args.force:
        print(f"[skip] {dataset} {ckpt} -> {output_dir}")
        return

    cmd = [
        sys.executable,
        "evaluate_objects.py",
        "--model_path",
        str(ckpt),
        "--dataset",
        dataset,
        "--output_root",
        str(output_dir),
        "--dataset_root",
        args.dataset_root,
        "--data_path",
        args.data_path,
        "--gpu",
        args.gpu,
        "--steps",
        str(args.diffusion_steps),
        "--k",
        str(args.k),
    ]
    if args.use_ema:
        cmd.append("--use_ema")
    if args.save_intermediate:
        cmd.append("--save_intermediate")

    print("[run] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_metric_rows(output_dir: Path, dataset: str, label: str, step: int) -> list[dict[str, object]]:
    import pandas as pd

    rows: list[dict[str, object]] = []
    for summary in summary_files(output_dir, dataset):
        df = pd.read_csv(summary, index_col=0)
        if df.empty:
            continue
        metrics = df.iloc[-1]
        setting = summary.parent.name
        match = SETTING_RE.match(setting)
        rows.append(
            {
                "step": step,
                "dataset": dataset,
                "setting": setting,
                "resolution": int(match.group("resolution")) if match else None,
                "noise": float(match.group("noise")) if match else None,
                "model": label,
                "cd": float(metrics.get("cd_sph(mean)", float("nan"))),
                "p2m": float(metrics.get("p2f(mean)", float("nan"))),
                "summary_path": str(summary),
            }
        )
    return rows


def compare_rows(rows: Iterable[dict[str, object]], baseline_name: str, candidate_name: str):
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    index_cols = ["step", "dataset", "setting", "resolution", "noise"]
    pivot = df.pivot_table(index=index_cols, columns="model", values=["cd", "p2m"], aggfunc="first")
    pivot.columns = [f"{model}_{metric}" for metric, model in pivot.columns]
    pivot = pivot.reset_index()

    base_cd = f"{baseline_name}_cd"
    cand_cd = f"{candidate_name}_cd"
    base_p2m = f"{baseline_name}_p2m"
    cand_p2m = f"{candidate_name}_p2m"
    if base_cd in pivot and cand_cd in pivot:
        pivot["delta_cd"] = pivot[cand_cd] - pivot[base_cd]
    if base_p2m in pivot and cand_p2m in pivot:
        pivot["delta_p2m"] = pivot[cand_p2m] - pivot[base_p2m]
    return pivot.sort_values(index_cols)


def print_comparison_table(comparison, comparison_path: Path) -> None:
    print(f"\n[comparison] {comparison_path}")
    if comparison.empty:
        print("No metrics found.")
        return
    display_cols = [col for col in comparison.columns if col not in {"setting"}]
    print(comparison[display_cols].to_string(index=False))


def main() -> None:
    args = parse_args()
    import pandas as pd

    save_root = Path(args.save_root)
    output_root = Path(args.output_root) if args.output_root else save_root / "manual_eval_objects"
    output_root.mkdir(parents=True, exist_ok=True)

    all_comparison_frames_by_dataset: dict[str, list[pd.DataFrame]] = {dataset: [] for dataset in args.datasets}
    pairs = find_pairs(args)

    for pair in pairs:
        for dataset in args.datasets:
            pair_rows: list[dict[str, object]] = []
            baseline_out = output_root / f"step_{pair.step}" / args.baseline_name / dataset
            candidate_out = output_root / f"step_{pair.step}" / pair.candidate_name / dataset

            run_evaluate_objects(pair.baseline, baseline_out, dataset, args)
            run_evaluate_objects(pair.candidate, candidate_out, dataset, args)

            pair_rows.extend(read_metric_rows(baseline_out, dataset, args.baseline_name, pair.step))
            pair_rows.extend(read_metric_rows(candidate_out, dataset, pair.candidate_name, pair.step))

            dataset_out = output_root / dataset
            dataset_out.mkdir(parents=True, exist_ok=True)

            raw_path = dataset_out / f"step_{pair.step}_{pair.candidate_name}_raw_metrics.csv"
            pd.DataFrame(pair_rows).to_csv(raw_path, index=False, float_format="%.12f")

            comparison = compare_rows(pair_rows, args.baseline_name, pair.candidate_name)
            comparison_path = dataset_out / f"step_{pair.step}_{pair.candidate_name}_comparison.csv"
            comparison.to_csv(comparison_path, index=False, float_format="%.12f")
            all_comparison_frames_by_dataset[dataset].append(comparison)
            print_comparison_table(comparison, comparison_path)

    for dataset, frames in all_comparison_frames_by_dataset.items():
        if not frames:
            continue
        all_path = output_root / dataset / "all_comparisons.csv"
        pd.concat(frames, ignore_index=True).to_csv(all_path, index=False, float_format="%.12f")
        print(f"\n[done] wrote {dataset} merged comparison: {all_path}")


if __name__ == "__main__":
    main()
