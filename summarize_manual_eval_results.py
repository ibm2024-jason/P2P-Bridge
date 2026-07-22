import argparse
import re
from pathlib import Path


COMPARISON_RE = re.compile(r"step_(?P<step>\d+)_(?P<candidate>.+)_comparison\.csv$")
RAW_RE = re.compile(r"step_(?P<step>\d+)_(?P<candidate>.+)_raw_metrics\.csv$")
SUMMARY_RE = re.compile(r"Summary_(?P<dataset>PUNet|PCNet)\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge manual PUNet/PCNet evaluation outputs into large CSV summary files."
    )
    parser.add_argument("--eval_root", default="experiments/rodr_compare_punet_full/manual_eval_objects")
    parser.add_argument("--out", default=None, help="Output directory. Defaults to <eval_root>/summary_csv.")
    parser.add_argument("--baseline_name", default="baseline_mse")
    return parser.parse_args()


def infer_dataset(path: Path) -> str | None:
    for part in path.parts:
        if part in {"PUNet", "PCNet"}:
            return part
    return None


def read_comparison_csv(path: Path):
    import pandas as pd

    match = COMPARISON_RE.match(path.name)
    df = pd.read_csv(path)
    df["source_file"] = str(path)
    df["candidate"] = match.group("candidate") if match else None
    if "step" not in df and match:
        df["step"] = int(match.group("step"))
    if "dataset" not in df:
        df["dataset"] = infer_dataset(path)
    return df


def read_raw_csv(path: Path):
    import pandas as pd

    match = RAW_RE.match(path.name)
    df = pd.read_csv(path)
    df["source_file"] = str(path)
    df["candidate"] = match.group("candidate") if match else None
    if "step" not in df and match:
        df["step"] = int(match.group("step"))
    if "dataset" not in df:
        df["dataset"] = infer_dataset(path)
    return df


def read_summary_csv(path: Path):
    import pandas as pd

    match = SUMMARY_RE.match(path.name)
    dataset = match.group("dataset") if match else infer_dataset(path)
    df = pd.read_csv(path, index_col=0).reset_index(names="model")
    df["dataset"] = dataset
    df["source_file"] = str(path)
    return df


def write_dataset_splits(df, out_dir: Path, name: str) -> None:
    if df.empty or "dataset" not in df:
        return
    for dataset, dataset_df in df.groupby("dataset", dropna=False):
        if not dataset:
            continue
        dataset_path = out_dir / f"{dataset}_{name}.csv"
        dataset_df.to_csv(dataset_path, index=False, float_format="%.12f")
        print(f"[write] {dataset_path}")


def summarize_comparisons(comparisons, baseline_name: str):
    import pandas as pd

    if comparisons.empty:
        return pd.DataFrame()

    metric_cols = [col for col in comparisons.columns if col.endswith("_cd") or col.endswith("_p2m")]
    group_cols = [col for col in ["dataset", "step", "candidate"] if col in comparisons.columns]
    summary = comparisons.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()

    candidate_names = [name for name in summary.get("candidate", pd.Series(dtype=str)).dropna().unique()]
    for candidate in candidate_names:
        base_cd = f"{baseline_name}_cd"
        cand_cd = f"{candidate}_cd"
        base_p2m = f"{baseline_name}_p2m"
        cand_p2m = f"{candidate}_p2m"
        if base_cd in summary and cand_cd in summary:
            summary[f"{candidate}_mean_delta_cd"] = summary[cand_cd] - summary[base_cd]
        if base_p2m in summary and cand_p2m in summary:
            summary[f"{candidate}_mean_delta_p2m"] = summary[cand_p2m] - summary[base_p2m]
    return summary


def main() -> None:
    args = parse_args()
    import pandas as pd

    eval_root = Path(args.eval_root)
    out_dir = Path(args.out) if args.out else eval_root / "summary_csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_frames = [read_comparison_csv(path) for path in sorted(eval_root.rglob("*_comparison.csv"))]
    raw_frames = [read_raw_csv(path) for path in sorted(eval_root.rglob("*_raw_metrics.csv"))]
    summary_frames = [read_summary_csv(path) for path in sorted(eval_root.rglob("Summary_*.csv"))]

    comparisons = pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()
    raw_metrics = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    summaries = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    mean_comparisons = summarize_comparisons(comparisons, args.baseline_name)

    outputs = {
        "all_comparisons_big.csv": comparisons,
        "all_raw_metrics_big.csv": raw_metrics,
        "all_summary_files_big.csv": summaries,
        "mean_comparison_by_dataset_step.csv": mean_comparisons,
    }

    for filename, df in outputs.items():
        path = out_dir / filename
        df.to_csv(path, index=False, float_format="%.12f")
        print(f"[write] {path} ({len(df)} rows)")

    write_dataset_splits(comparisons, out_dir, "comparisons_big")
    write_dataset_splits(raw_metrics, out_dir, "raw_metrics_big")
    write_dataset_splits(mean_comparisons, out_dir, "mean_comparison")

    if comparisons.empty and raw_metrics.empty and summaries.empty:
        print(f"[warn] No CSV inputs found under {eval_root}")


if __name__ == "__main__":
    main()
