from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SETTING_RE = re.compile(r".*_steps_(?P<steps>\d+)_(?P<resolution>\d+)_(?P<noise>[0-9.]+)$")


@dataclass(frozen=True)
class PredictionSetting:
    step: int
    model: str
    dataset: str
    setting: str
    resolution: int
    noise: float
    pcl_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Uniformity, LMFR, and Normal Consistency on existing "
            "manual_eval_objects predictions, then compare baseline and RODR."
        )
    )
    parser.add_argument("--eval_root", default="experiments/rodr_compare_punet_full/manual_eval_objects")
    parser.add_argument("--dataset_root", default="./data/objects")
    parser.add_argument("--out", default=None, help="Output directory. Defaults to <eval_root>/nc_summary.")
    parser.add_argument("--baseline_name", default="baseline_mse")
    parser.add_argument("--candidate_glob", default="rodr_w*")
    parser.add_argument("--datasets", nargs="+", default=["PUNet", "PCNet"], choices=["PUNet", "PCNet"])
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max_shapes", type=int, default=None)
    return parser.parse_args()


def parse_setting_name(setting: str) -> tuple[int, float] | None:
    match = SETTING_RE.match(setting)
    if not match:
        return None
    return int(match.group("resolution")), float(match.group("noise"))


def parse_step(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("step_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                return None
    return None


def discover_prediction_settings(args: argparse.Namespace) -> list[PredictionSetting]:
    eval_root = Path(args.eval_root)
    model_names = [args.baseline_name] + sorted(
        path.name for path in eval_root.glob(args.candidate_glob) if path.is_dir()
    )
    settings: list[PredictionSetting] = []

    for pcl_dir in sorted(eval_root.rglob("pcl")):
        if not pcl_dir.is_dir():
            continue
        step = parse_step(pcl_dir)
        if step is None or (args.step is not None and step != args.step):
            continue

        parts = pcl_dir.parts
        model = next((part for part in parts if part in model_names), None)
        dataset = next((part for part in parts if part in args.datasets), None)
        setting = pcl_dir.parent.name
        parsed = parse_setting_name(setting)
        if model is None or dataset is None or parsed is None:
            continue

        resolution, noise = parsed
        settings.append(
            PredictionSetting(
                step=step,
                model=model,
                dataset=dataset,
                setting=setting,
                resolution=resolution,
                noise=noise,
                pcl_dir=pcl_dir,
            )
        )

    if not settings:
        raise FileNotFoundError(f"No prediction pcl dirs found under {eval_root}")
    return settings


def load_xyz(path: Path) -> torch.Tensor:
    import torch

    return torch.from_numpy(np.loadtxt(path, dtype=np.float32))[:, :3]


def load_meshes(mesh_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    import point_cloud_utils as pcu

    meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(mesh_dir.glob("*.off")):
        verts, faces = pcu.load_mesh_vf(path)
        meshes[path.stem] = (verts.astype(np.float64), faces.astype(np.int32))
    return meshes


def compute_uniformity(pcl: torch.Tensor, k: int) -> float:
    import torch
    from pytorch3d.ops import knn_points

    dist, _, _ = knn_points(pcl, pcl, K=k + 1)
    dist = torch.sqrt(dist[:, :, 1:] + 1e-8)
    mean = torch.mean(dist, dim=-1)
    std = torch.std(dist, dim=-1)
    return torch.mean(std / (mean + 1e-8)).item()


def compute_lmfr(pcl: torch.Tensor, k: int) -> float:
    import torch
    from pytorch3d.ops import knn_points

    _, _, nns = knn_points(pcl, pcl, K=k, return_nn=True)
    mean = torch.mean(nns, dim=2, keepdim=True)
    centered = nns - mean
    cov = torch.matmul(centered.transpose(-1, -2), centered) / max(k - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(cov)
    lmfr = torch.sqrt(torch.clamp(eigenvalues[..., 0], min=1e-9))
    return torch.mean(lmfr).item()


def estimate_normals_pca(pcl: torch.Tensor, k: int) -> torch.Tensor:
    import torch
    from pytorch3d.ops import knn_points

    _, _, nns = knn_points(pcl, pcl, K=k, return_nn=True)
    mean = torch.mean(nns, dim=2, keepdim=True)
    centered = nns - mean
    cov = torch.matmul(centered.transpose(-1, -2), centered)
    _, eigenvectors = torch.linalg.eigh(cov)
    normals = eigenvectors[:, :, :, 0]
    return normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)


def compute_normal_consistency(pcl: torch.Tensor, verts: np.ndarray, faces: np.ndarray, k: int) -> float:
    import point_cloud_utils as pcu

    pcl_np = pcl.detach().cpu().numpy().astype(np.float64)
    normals = estimate_normals_pca(pcl.unsqueeze(0), k=k).squeeze(0).detach().cpu().numpy().astype(np.float64)

    _, face_ids, _ = pcu.closest_points_on_mesh(pcl_np, verts, faces)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    gt_normals = face_normals[face_ids]

    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    gt_normals = gt_normals / (np.linalg.norm(gt_normals, axis=1, keepdims=True) + 1e-8)
    return float(np.clip(np.abs(np.sum(normals * gt_normals, axis=1)), 0.0, 1.0).mean())


def shape_gt_name(pred_name: str) -> str:
    return pred_name[:-6] if pred_name.endswith("_noisy") else pred_name


def evaluate_setting(setting: PredictionSetting, meshes: dict[str, tuple[np.ndarray, np.ndarray]], args: argparse.Namespace):
    import torch
    from tqdm.auto import tqdm

    device = torch.device(args.gpu)
    rows: list[dict[str, object]] = []
    xyz_files = sorted(setting.pcl_dir.glob("*.xyz"))
    if args.max_shapes is not None:
        xyz_files = xyz_files[: args.max_shapes]

    for xyz_path in tqdm(xyz_files, desc=f"{setting.model} {setting.dataset} {setting.setting}"):
        name = xyz_path.stem
        gt_name = shape_gt_name(name)
        if gt_name not in meshes:
            continue

        pcl = load_xyz(xyz_path).to(device)
        pcl_batch = pcl.unsqueeze(0)
        verts, faces = meshes[gt_name]
        rows.append(
            {
                "step": setting.step,
                "dataset": setting.dataset,
                "setting": setting.setting,
                "resolution": setting.resolution,
                "noise": setting.noise,
                "model": setting.model,
                "shape": name,
                "unif": compute_uniformity(pcl_batch, args.k),
                "lmfr": compute_lmfr(pcl_batch, args.k),
                "nc": compute_normal_consistency(pcl, verts, faces, args.k),
                "pcl_path": str(xyz_path),
            }
        )
    return rows


def build_comparison(means: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    import pandas as pd

    index_cols = ["step", "dataset", "setting", "resolution", "noise"]
    pivot = means.pivot_table(index=index_cols, columns="model", values=["unif", "lmfr", "nc"], aggfunc="first")
    pivot.columns = [f"{model}_{metric}" for metric, model in pivot.columns]
    pivot = pivot.reset_index()

    candidates = sorted(
        {
            col.rsplit("_", 1)[0]
            for col in pivot.columns
            if col.endswith("_unif") and col != f"{baseline_name}_unif"
        }
    )
    for candidate in candidates:
        for metric in ["unif", "lmfr"]:
            base_col = f"{baseline_name}_{metric}"
            cand_col = f"{candidate}_{metric}"
            if base_col in pivot and cand_col in pivot:
                pivot[f"delta_{metric}"] = pivot[cand_col] - pivot[base_col]
                pivot[f"improve_{metric}_pct"] = (pivot[base_col] - pivot[cand_col]) / pivot[base_col] * 100
        base_col = f"{baseline_name}_nc"
        cand_col = f"{candidate}_nc"
        if base_col in pivot and cand_col in pivot:
            pivot["delta_nc"] = pivot[cand_col] - pivot[base_col]
            pivot["improve_nc_pct"] = (pivot[cand_col] - pivot[base_col]) / pivot[base_col] * 100

    return pivot.sort_values(index_cols)


def main() -> None:
    args = parse_args()

    import pandas as pd
    import torch

    torch.cuda.set_device(int(args.gpu.split(":", 1)[1]) if args.gpu.startswith("cuda:") else 0)

    out_dir = Path(args.out) if args.out else Path(args.eval_root) / "nc_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = discover_prediction_settings(args)
    mesh_cache: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    shape_rows: list[dict[str, object]] = []

    for setting in settings:
        if setting.dataset not in mesh_cache:
            mesh_cache[setting.dataset] = load_meshes(Path(args.dataset_root) / setting.dataset / "meshes" / "test")
        shape_rows.extend(evaluate_setting(setting, mesh_cache[setting.dataset], args))

    shape_df = pd.DataFrame(shape_rows)
    shape_path = out_dir / "nc_shape_metrics.csv"
    shape_df.to_csv(shape_path, index=False, float_format="%.12f")
    print(f"[write] {shape_path} ({len(shape_df)} rows)")

    if shape_df.empty:
        print("[warn] No shape metrics computed.")
        return

    mean_cols = ["step", "dataset", "setting", "resolution", "noise", "model"]
    means = shape_df.groupby(mean_cols, dropna=False)[["unif", "lmfr", "nc"]].mean().reset_index()
    means_path = out_dir / "nc_mean_metrics.csv"
    means.to_csv(means_path, index=False, float_format="%.12f")
    print(f"[write] {means_path} ({len(means)} rows)")

    comparison = build_comparison(means, args.baseline_name)
    comparison_path = out_dir / "nc_comparison.csv"
    comparison.to_csv(comparison_path, index=False, float_format="%.12f")
    print(f"[write] {comparison_path} ({len(comparison)} rows)")

    for dataset, dataset_df in comparison.groupby("dataset", dropna=False):
        path = out_dir / f"{dataset}_nc_comparison.csv"
        dataset_df.to_csv(path, index=False, float_format="%.12f")
        print(f"[write] {path}")

    print("\n[comparison]")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
