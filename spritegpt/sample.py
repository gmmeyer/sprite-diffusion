"""Generate a sample grid from a checkpoint (uses EMA weights).

    uv run python -m spritegpt.sample --ckpt runs/m1-uncond32/ckpt_latest.pt --out grid.png
    uv run python -m spritegpt.sample --ckpt runs/m3-text64/ckpt_latest.pt \
        --prompt "a blue ghost sprite" --guidance 5 --out ghost.png
"""

import argparse

import torch

from spritegpt.flow import sample
from spritegpt.unet import UNet
from spritegpt.utils import save_image_grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="grid.png")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--class-id", type=int, default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--raw-weights", action="store_true", help="use raw weights instead of EMA")
    args = ap.parse_args()

    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "config" in ck:  # HF export: EMA weights + config, no optimizer/raw weights
        cfg, step = ck["config"], ck["config"]["train_steps"]
        num_classes, ctx_dim = cfg["num_classes"], cfg.get("ctx_dim")
        state = ck["ema_state_dict"]
    else:  # training checkpoint
        ta, cfg, step = ck["args"], ck["args"], ck["step"]
        num_classes = None
        if ta.get("class_cond"):
            # class embedding table size minus the null token
            num_classes = ck["model"]["class_emb.weight"].shape[0] - 1
        ctx_dim = 512 if ta.get("text_data") is not None else None
        state = ck["model"] if args.raw_weights else ck["ema"]["shadow"]
    text_cond = ctx_dim is not None
    model = UNet(
        img_size=cfg["img_size"],
        base=cfg["base"],
        ch_mult=tuple(int(c) for c in cfg["ch_mult"].split(",")),
        num_res=cfg["num_res"],
        attn_res=tuple(int(r) for r in cfg["attn_res"].split(",")),
        dropout=0.0,
        num_classes=num_classes,
        ctx_dim=ctx_dim,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    y, ctx, mask, nctx, nmask, guidance = None, None, None, None, None, 0.0
    if text_cond:
        if args.prompt is None:
            raise SystemExit("this checkpoint is text-conditioned: pass --prompt")
        from spritegpt.text import embed_prompts

        emb, m = embed_prompts([args.prompt, ""], device)
        ctx = emb[0:1].expand(args.n, -1, -1)
        mask = m[0:1].expand(args.n, -1)
        nctx, nmask = emb[1:2], m[1:2]
        guidance = args.guidance
    elif num_classes is not None and args.class_id is not None:
        y = torch.full((args.n,), args.class_id, device=device, dtype=torch.long)
        guidance = args.guidance

    gen = None
    if args.seed is not None:
        gen = torch.Generator(device=device).manual_seed(args.seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        imgs = sample(model, args.n, cfg["img_size"], steps=args.steps, y=y,
                      ctx=ctx, ctx_mask=mask, null_ctx=nctx, null_mask=nmask,
                      guidance=guidance, num_classes=num_classes,
                      device=device, generator=gen)
    save_image_grid(imgs.float(), args.out)
    print(f"saved {args.n} samples (step {step}) -> {args.out}")


if __name__ == "__main__":
    main()
