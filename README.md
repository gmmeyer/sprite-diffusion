# spritegpt

Small pixel-space diffusion model for emoji and game sprites, trained with
rectified flow (flow matching) on an RTX 5090.

## Data

- **OpenMoji** (CC-BY-SA 4.0) — ~4.5k emoji, 72x72 color PNGs
- **Twemoji** (CC-BY 4.0, jdecked fork) — ~4k emoji, 72x72 PNGs
- Composited onto white (RGBA alpha is awkward to diffuse directly), resized
  to 32x32, packed into `data/emoji32.npz` with category labels from
  `openmoji.json` for class conditioning.

```sh
uv run python scripts/prepare_data.py --size 32
```

## Model

ADM-style U-Net (~29M params at 32px): base 128, channel mult (1,2,2),
2 res blocks per level, self-attention at 16x16 and 8x8, scale-shift
GroupNorm conditioning. Rectified flow objective (`x_t = (1-t)x0 + t*eps`,
predict `v = eps - x0`), logit-normal timestep sampling, EMA weights
(decay 0.9995), Euler ODE sampler.

## Train

```sh
# milestone 1: unconditional 32px
uv run python -m spritegpt.train --data data/emoji32.npz --run-name m1-uncond32

# milestone 2: class-conditional (13 emoji groups), CFG cond-drop 10%
uv run python -m spritegpt.train --data data/emoji32.npz --run-name m2-cond32 --class-cond
```

Loss log: `runs/<name>/log.csv`. Sample grids every 2k steps in
`runs/<name>/samples/`. Resume with `--resume runs/<name>/ckpt_latest.pt`.

## Sample

```sh
uv run python -m spritegpt.sample --ckpt runs/m1-uncond32/ckpt_latest.pt --out grid.png
# class-conditional: --class-id 9 --guidance 3.0  (9 = smileys-emotion)
```

## Text-conditioned sampling (milestone 3 model)

```sh
uv run python scripts/make_captions.py   # captions + cached CLIP embeddings
uv run python -m spritegpt.train --data data/combined64.npz --text-data data/text64.npz \
    --run-name m3-text64 --img-size 64 --ch-mult 1,2,2,3 --batch-size 128 --steps 120000
uv run python -m spritegpt.sample --ckpt runs/m3-text64/ckpt_latest.pt \
    --prompt "a blue ghost sprite" --guidance 5 --out ghost.png
```

Trained weights: https://huggingface.co/gmmeyer/sprite-gpt

## Milestones

1. [x] Unconditional 32px emoji
2. [x] Class-conditional 64px (13 emoji groups + 18 sprite packs, 31 classes)
3. [x] CLIP text conditioning via cross-attention (emoji names are free captions;
       composes unseen attribute combinations like "a purple dragon emoji")
4. [ ] Stretch: 4-frame walk-cycle sprite sheets
