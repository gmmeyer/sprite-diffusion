"""Export distribution-ready weights (EMA only, no optimizer state) for HF upload.

    uv run python scripts/export_hf.py --ckpt runs/m2-cond64/ckpt_latest.pt \
        --out hf_export/sprite-gpt-cond64.pt
"""

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ta = ck["args"]
    num_classes = None
    if ta.get("class_cond"):
        num_classes = ck["model"]["class_emb.weight"].shape[0] - 1
    config = {
        "img_size": ta["img_size"],
        "base": ta["base"],
        "ch_mult": ta["ch_mult"],
        "num_res": ta["num_res"],
        "attn_res": ta["attn_res"],
        "num_classes": num_classes,
        "ctx_dim": 512 if ta.get("text_data") else None,
        "objective": "rectified_flow",
        "train_steps": ck["step"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "ema_state_dict": ck["ema"]["shadow"]}, args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"exported {args.out} ({size_mb:.0f}MB) config={config}")


if __name__ == "__main__":
    main()
