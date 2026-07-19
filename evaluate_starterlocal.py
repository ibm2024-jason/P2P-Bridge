import argparse
import json
import os
from pathlib import Path

import torch
from loguru import logger
from omegaconf import OmegaConf

from dataloaders.dataloader import get_dataloader
from models.evaluation import evaluate
from models.model_loader import load_diffusion
from models.train_utils import set_seed, setup_output_subdirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to a P2P-Bridge checkpoint.")
    parser.add_argument("--output_dir", default=None, help="Directory for visualizations and metrics.")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--steps", type=int, default=None, help="Override sampling timesteps.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--accum_iter", type=int, default=8, help="Number of validation batches to evaluate.")
    parser.add_argument("--fast", action="store_true", help="Use faster approximate metric path when available.")
    parser.add_argument("--distribution_type", default="none")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_dir = Path(args.model_path).resolve().parent
    cfg = OmegaConf.load(ckpt_dir / "opt.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.create(vars(args)))

    cfg.model_path = str(Path(args.model_path).resolve())
    cfg.restart = False
    cfg.local_rank = 0
    cfg.global_rank = 0
    cfg.global_size = 1
    cfg.gpu = 0
    cfg.distribution_type = "none"
    cfg.sampling.bs = args.batch_size
    cfg.sampling.accum_iter = args.accum_iter
    cfg.data.workers = args.workers
    if args.steps is not None:
        cfg.diffusion.sampling_timesteps = args.steps

    if args.output_dir is None:
        cfg.output_dir = str(ckpt_dir / "starterlocal_eval")
    else:
        cfg.output_dir = args.output_dir
    (outf_syn,) = setup_output_subdirs(cfg.output_dir, "output")
    cfg.outf_syn = outf_syn

    set_seed(cfg)
    torch.cuda.set_device(0)
    model, _ = load_diffusion(cfg)
    model.eval()

    _, val_loader, _, _ = get_dataloader(cfg, sampling=True)
    metrics = evaluate(model, val_loader, cfg, step=0, sampling=False, save_npy=True, fast=args.fast)

    metrics_path = os.path.join(cfg.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to {}", metrics_path)
    logger.info(metrics)


if __name__ == "__main__":
    main()
