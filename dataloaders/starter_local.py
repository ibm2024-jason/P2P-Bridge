import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class StarterLocalPatchDataset(Dataset):
    """Starter-code dataset.

    For train/validate/test splits with datalist entries, clean points are sampled
    from dataset_train meshes and noisy points are generated online. For local_*
    splits, paired local clean/noisy npy files are used for validation.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        patch_size: int = 2048,
        num_patches: int = 1000,
        dense_size: int = 50000,
        noise_min: float = 0.005,
        noise_max: float = 0.020,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.clean_dir = self.root / "local" / "clean"
        self.noisy_dir = self.root / "local" / "noisy"
        self.mesh_root = self.root / "dataset_train"
        self.cache_root = self.root / "cache" / "starter_mesh_points"
        self.patch_size = int(patch_size)
        self.num_patches = int(num_patches)
        self.dense_size = int(dense_size)
        self.noise_min = float(noise_min)
        self.noise_max = float(noise_max)
        self.seed = int(seed)
        self.use_local_pairs = split.startswith("local_")
        self.cache_root.mkdir(parents=True, exist_ok=True)

        all_local_names = [p.stem for p in sorted(self.clean_dir.glob("*.npy"))]
        list_path = self.root / "datalist" / f"{split}.txt"
        if self.use_local_pairs:
            pivot = max(1, int(0.8 * len(all_local_names)))
            if split == "local_train":
                names = all_local_names[:pivot]
            else:
                names = all_local_names[pivot:]
        elif list_path.exists():
            names = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
        else:
            names = all_local_names

        if self.use_local_pairs:
            self.names = [
                name
                for name in names
                if (self.clean_dir / f"{name}.npy").exists() and (self.noisy_dir / f"{name}.npy").exists()
            ]
            if not self.names:
                raise RuntimeError(f"No paired starter local npy files found for split={split!r} under {root}")
        else:
            self.names = [name for name in names if self._mesh_path(name).exists()]
            if not self.names:
                raise RuntimeError(f"No dataset_train meshes found for split={split!r} under {root}")

    @staticmethod
    def _rel_to_local_name(rel: str) -> str:
        parts = rel.replace("\\", "/").split("/")
        if len(parts) >= 3 and parts[0] == "shapenet":
            return f"{parts[1]}_{parts[2]}"
        return Path(rel).stem

    def __len__(self) -> int:
        return len(self.names) * self.num_patches

    def _mesh_path(self, rel: str) -> Path:
        return self.mesh_root / rel / "models" / "model_normalized.obj"

    def _cache_path(self, rel: str) -> Path:
        safe = rel.replace("/", "_")
        digest = hashlib.md5(rel.encode()).hexdigest()[:10]
        return self.cache_root / f"{safe}_{digest}_n{self.dense_size}.npy"

    def _load_mesh_points(self, rel: str, rng: np.random.Generator) -> torch.Tensor:
        cache_path = self._cache_path(rel)
        if cache_path.exists():
            points = np.load(cache_path).astype(np.float32)
            if points.shape[0] >= self.dense_size:
                return torch.from_numpy(points)

        import trimesh

        mesh = trimesh.load(self._mesh_path(rel), force="mesh", skip_materials=True, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        points, _ = trimesh.sample.sample_surface(mesh, self.dense_size)
        points = np.asarray(points, dtype=np.float32)
        rng.shuffle(points)
        np.save(cache_path, points)
        return torch.from_numpy(points)

    def _load_pair(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        clean = np.load(self.clean_dir / f"{name}.npy").astype(np.float32)
        noisy = np.load(self.noisy_dir / f"{name}.npy").astype(np.float32)
        return torch.from_numpy(clean), torch.from_numpy(noisy)

    def __getitem__(self, idx: int) -> dict:
        cloud_idx = idx % len(self.names)
        generator = torch.Generator()
        generator.manual_seed(self.seed + idx * 1000003)

        if self.use_local_pairs:
            clean, noisy = self._load_pair(self.names[cloud_idx])
        else:
            rng = np.random.default_rng(self.seed + idx * 1000003)
            clean = self._load_mesh_points(self.names[cloud_idx], rng)
            noise_std = float(rng.uniform(self.noise_min, self.noise_max))
            noisy = clean + torch.randn(clean.shape, generator=generator, dtype=clean.dtype) * noise_std

        n_points = min(clean.shape[0], noisy.shape[0])
        patch_size = min(self.patch_size, n_points)

        patch_idx = torch.randperm(n_points, generator=generator)[:patch_size]

        clean_patch = clean[patch_idx].clone()
        noisy_patch = noisy[patch_idx].clone()

        center = clean_patch.mean(dim=0)
        clean_patch = clean_patch - center
        noisy_patch = noisy_patch - center
        scale = torch.max(torch.norm(noisy_patch, dim=1)).clamp_min(1e-8)
        clean_patch = clean_patch / scale
        noisy_patch = noisy_patch / scale

        return {
            "noisy_points": noisy_patch,
            "clean_points": clean_patch,
            "center": center,
            "scale": scale,
            "name": self.names[cloud_idx],
        }
