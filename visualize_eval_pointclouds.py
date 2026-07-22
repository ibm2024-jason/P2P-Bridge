import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render high-resolution 3D comparison images from existing .xyz predictions. "
            "Colors show nearest-neighbor distance to the clean GT point cloud."
        )
    )
    parser.add_argument("--eval_root", default="experiments/rodr_compare_punet_full/manual_eval_objects")
    parser.add_argument("--dataset_root", default="./data/objects")
    parser.add_argument("--step", type=int, default=300000)
    parser.add_argument("--datasets", nargs="+", default=["PUNet"], choices=["PUNet", "PCNet"])
    parser.add_argument("--resolutions", nargs="+", type=int, default=[10000])
    parser.add_argument("--noises", nargs="+", type=float, default=[0.03])
    parser.add_argument("--models", nargs="+", default=["baseline_mse", "rodr_w1.0"])
    parser.add_argument("--shape_names", nargs="+", default=None, help="Shape names without .xyz. Auto-select if omitted.")
    parser.add_argument("--max_shapes", type=int, default=4)
    parser.add_argument("--max_points", type=int, default=12000, help="Subsample each cloud for clearer rendering.")
    parser.add_argument("--diffusion_steps", type=int, default=5)
    parser.add_argument("--out_dir", default="figures/eval_pointclouds")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--point_size", type=float, default=0.18)
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument(
        "--normalize",
        choices=["shape", "figure"],
        default="shape",
        help="Normalize color range per shape row or over the whole figure.",
    )
    parser.add_argument("--no_pdf", action="store_true", help="Only write PNG.")
    return parser.parse_args()


def noise_token(noise: float) -> str:
    return f"{noise:g}"


def load_xyz(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    points = np.loadtxt(path, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected Nx3 .xyz file, got {points.shape} from {path}")
    return points[:, :3]


def gt_path(dataset_root: Path, dataset: str, resolution: int, shape_name: str) -> Path:
    return dataset_root / dataset / "pointclouds" / "test" / f"{resolution}_poisson" / f"{shape_name}.xyz"


def noisy_path(dataset_root: Path, dataset: str, resolution: int, noise: float, shape_name: str) -> Path:
    setting = f"{dataset}_{resolution}_poisson_{noise_token(noise)}"
    return dataset_root / "examples" / setting / f"{shape_name}.xyz"


def pred_path(
    eval_root: Path,
    step: int,
    model_name: str,
    dataset: str,
    diffusion_steps: int,
    resolution: int,
    noise: float,
    shape_name: str,
) -> Path:
    setting = f"P2P-Bridge_steps_{diffusion_steps}_{resolution}_{noise_token(noise)}"
    return eval_root / f"step_{step}" / model_name / dataset / dataset / setting / "pcl" / f"{shape_name}.xyz"


def available_shapes(
    eval_root: Path,
    dataset_root: Path,
    step: int,
    models: list[str],
    dataset: str,
    diffusion_steps: int,
    resolution: int,
    noise: float,
) -> list[str]:
    shape_sets: list[set[str]] = []
    noisy_dir = dataset_root / "examples" / f"{dataset}_{resolution}_poisson_{noise_token(noise)}"
    if noisy_dir.exists():
        shape_sets.append({path.stem for path in noisy_dir.glob("*.xyz")})
    gt_dir = dataset_root / dataset / "pointclouds" / "test" / f"{resolution}_poisson"
    if gt_dir.exists():
        shape_sets.append({path.stem for path in gt_dir.glob("*.xyz")})

    for model_name in models:
        setting = f"P2P-Bridge_steps_{diffusion_steps}_{resolution}_{noise_token(noise)}"
        pcl_dir = eval_root / f"step_{step}" / model_name / dataset / dataset / setting / "pcl"
        if pcl_dir.exists():
            shape_sets.append({path.stem for path in pcl_dir.glob("*.xyz")})

    if not shape_sets:
        return []
    return sorted(set.intersection(*shape_sets))


def nn_distance_to_gt(points: np.ndarray, gt_points: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(gt_points).query(points, k=1, workers=-1)
    return distances.astype(np.float32)


def downsample(points: np.ndarray, values: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, values
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx], values[idx]


def set_equal_3d(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    radius = max(maxs - mins) / 2
    radius = max(radius, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def render_one(
    args: argparse.Namespace,
    dataset: str,
    resolution: int,
    noise: float,
    shape_names: list[str],
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.cm import ScalarMappable

    eval_root = Path(args.eval_root)
    dataset_root = Path(args.dataset_root)
    columns = ["Noisy"] + args.models
    loaded: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    all_values: list[np.ndarray] = []

    for row_idx, shape_name in enumerate(shape_names):
        gt = load_xyz(gt_path(dataset_root, dataset, resolution, shape_name))
        for col_idx, column in enumerate(columns):
            path = (
                noisy_path(dataset_root, dataset, resolution, noise, shape_name)
                if column == "Noisy"
                else pred_path(
                    eval_root,
                    args.step,
                    column,
                    dataset,
                    args.diffusion_steps,
                    resolution,
                    noise,
                    shape_name,
                )
            )
            points = load_xyz(path)
            values = nn_distance_to_gt(points, gt)
            points, values = downsample(points, values, args.max_points, seed=row_idx * 1009 + col_idx)
            loaded[(shape_name, column)] = (points, values)
            all_values.append(values)

    fig_width = max(4.0 * len(columns), 8.0)
    fig_height = max(3.6 * len(shape_names), 4.0)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=args.dpi)
    fig.patch.set_facecolor("white")

    if args.normalize == "figure":
        merged = np.concatenate(all_values)
        global_vmax = float(np.percentile(merged, 95))
    else:
        global_vmax = None

    last_scatter = None
    for row_idx, shape_name in enumerate(shape_names):
        row_values = [loaded[(shape_name, column)][1] for column in columns]
        row_vmax = global_vmax if global_vmax is not None else float(np.percentile(np.concatenate(row_values), 95))
        norm = colors.Normalize(vmin=0.0, vmax=max(row_vmax, 1e-8))

        for col_idx, column in enumerate(columns):
            ax = fig.add_subplot(len(shape_names), len(columns), row_idx * len(columns) + col_idx + 1, projection="3d")
            points, values = loaded[(shape_name, column)]
            order = np.argsort(values)
            points = points[order]
            values = values[order]
            last_scatter = ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=values,
                cmap=args.cmap,
                norm=norm,
                s=args.point_size,
                linewidths=0,
                alpha=0.95,
                rasterized=True,
            )
            set_equal_3d(ax, points)
            ax.view_init(elev=args.elev, azim=args.azim)
            ax.set_axis_off()
            if row_idx == 0:
                ax.set_title(column, fontsize=13, pad=0)
            if col_idx == 0:
                ax.text2D(-0.04, 0.5, shape_name, transform=ax.transAxes, rotation=90, va="center", fontsize=10)

    title = f"{dataset} step {args.step} | {resolution} points | noise {noise_token(noise)}"
    fig.suptitle(title, fontsize=15, y=0.995)
    if last_scatter is not None:
        cbar = fig.colorbar(
            ScalarMappable(norm=last_scatter.norm, cmap=args.cmap),
            ax=fig.axes,
            fraction=0.018,
            pad=0.01,
            shrink=0.82,
        )
        cbar.set_label("Nearest-neighbor distance to GT (darker=cleaner, brighter=noisier)", fontsize=10)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{dataset}_step{args.step}_{resolution}_noise{noise_token(noise).replace('.', 'p')}"
    png_path = out_dir / f"{base_name}.png"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    if not args.no_pdf:
        fig.savefig(out_dir / f"{base_name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    dataset_root = Path(args.dataset_root)

    written: list[Path] = []
    for dataset in args.datasets:
        for resolution in args.resolutions:
            for noise in args.noises:
                shapes = args.shape_names
                if shapes is None:
                    shapes = available_shapes(
                        eval_root,
                        dataset_root,
                        args.step,
                        args.models,
                        dataset,
                        args.diffusion_steps,
                        resolution,
                        noise,
                    )
                shapes = list(shapes)[: args.max_shapes]
                if not shapes:
                    print(f"[skip] No common .xyz shapes found for {dataset} {resolution} noise={noise_token(noise)}")
                    continue
                try:
                    out_path = render_one(args, dataset, resolution, noise, shapes)
                except FileNotFoundError as exc:
                    print(f"[skip] Missing file for {dataset} {resolution} noise={noise_token(noise)}: {exc}")
                    continue
                written.append(out_path)
                print(f"[write] {out_path}")

    if not written:
        raise SystemExit(
            "No figures were written. Check --eval_root, --dataset_root, --step, --models, and whether .xyz files exist."
        )


if __name__ == "__main__":
    main()
