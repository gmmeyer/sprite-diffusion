"""Train a rectified-flow U-Net on packed emoji/sprite images.

Milestone 1 (unconditional 32px):
    uv run python -m spritegpt.train --data data/emoji32.npz --run-name m1-uncond32

Class-conditional (milestone 2): add --class-cond
"""

import argparse
import copy
import csv
import math
import os
import time
from pathlib import Path

import torch

from spritegpt.data import GpuImageDataset
from spritegpt.ema import EMA
from spritegpt.flow import rf_loss, sample
from spritegpt.unet import UNet
from spritegpt.utils import save_image_grid


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--img-size", type=int, default=32)
    ap.add_argument("--base", type=int, default=128)
    ap.add_argument("--ch-mult", default="1,2,2")
    ap.add_argument("--num-res", type=int, default=2)
    ap.add_argument("--attn-res", default="16,8")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--class-cond", action="store_true")
    ap.add_argument("--cond-drop", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--ema-decay", type=float, default=0.9995)
    ap.add_argument("--no-hflip", action="store_true")
    ap.add_argument("--time-sampling", default="logit_normal", choices=["logit_normal", "uniform"])
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=3.0, help="CFG scale for preview grids (class-cond only)")
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def lr_at(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    frac = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac)))


def save_ckpt(path: Path, model, ema, opt, step: int, args) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {"step": step, "model": model.state_dict(), "ema": ema.state_dict(),
         "opt": opt.state_dict(), "args": vars(args)},
        tmp,
    )
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"

    run_dir = Path("runs") / args.run_name
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)

    ds = GpuImageDataset(args.data, device=device, hflip=not args.no_hflip)
    num_classes = ds.num_classes if args.class_cond else None
    model = UNet(
        img_size=args.img_size,
        base=args.base,
        ch_mult=tuple(int(c) for c in args.ch_mult.split(",")),
        num_res=args.num_res,
        attn_res=tuple(int(r) for r in args.attn_res.split(",")),
        dropout=args.dropout,
        num_classes=num_classes,
    ).to(device, memory_format=torch.channels_last)
    ema_model = copy.deepcopy(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"dataset: {len(ds)} images | model: {n_params/1e6:.1f}M params "
          f"| classes: {num_classes}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    ema = EMA(model, decay=args.ema_decay)

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"resumed from {args.resume} at step {start_step}", flush=True)

    log_path = run_dir / "log.csv"
    if not log_path.exists():
        log_path.write_text("step,loss,lr,imgs_per_sec\n")

    loss_acc, t0 = 0.0, time.perf_counter()
    for step in range(start_step, args.steps):
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = ds.batch(args.batch_size)
        x = x.contiguous(memory_format=torch.channels_last)
        if num_classes is not None:
            drop = torch.rand(y.shape[0], device=device) < args.cond_drop
            y = torch.where(drop, torch.full_like(y, num_classes), y)
        else:
            y = None

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = rf_loss(model, x, y, time_sampling=args.time_sampling)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        loss_acc += loss.item()

        if (step + 1) % args.log_every == 0:
            dt = time.perf_counter() - t0
            ips = args.log_every * args.batch_size / dt
            avg = loss_acc / args.log_every
            print(f"step {step+1:>7}/{args.steps} loss {avg:.4f} lr {lr:.2e} {ips:,.0f} imgs/s", flush=True)
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step + 1, f"{avg:.5f}", f"{lr:.2e}", f"{ips:.0f}"])
            loss_acc, t0 = 0.0, time.perf_counter()

        if (step + 1) % args.sample_every == 0 or step + 1 == args.steps:
            ema.copy_to(ema_model)
            ema_model.eval()
            gen = torch.Generator(device=device).manual_seed(42)
            if num_classes is not None:
                sy = torch.arange(64, device=device) % num_classes
                gy, gscale = sy, args.guidance
            else:
                gy, gscale = None, 0.0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                imgs = sample(ema_model, 64, args.img_size, steps=args.sample_steps,
                              y=gy, guidance=gscale, num_classes=num_classes,
                              device=device, generator=gen)
            save_image_grid(imgs.float(), run_dir / "samples" / f"step_{step+1:06d}.png", nrow=8)
            t0 = time.perf_counter()  # don't count sampling in imgs/s

        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            save_ckpt(run_dir / "ckpt_latest.pt", model, ema, opt, step + 1, args)
            if (step + 1) % 20000 == 0:
                save_ckpt(run_dir / f"ckpt_{step+1:06d}.pt", model, ema, opt, step + 1, args)
            t0 = time.perf_counter()

    print("done", flush=True)


if __name__ == "__main__":
    main()
