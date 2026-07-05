"""Derive a caption per image in a packed npz, then embed with frozen CLIP.

Emoji get their official Unicode/OpenMoji annotations ("grinning face with
sweat"); sprites get templated captions from filenames where meaningful
(DCSS, DawnLike) or pack-level descriptions (Kenney tile packs).

Writes a sidecar npz aligned index-for-index with the input:
  captions   [N] str
  ctx        [N, L, 512] fp16   CLIP ViT-B/32 text token embeddings
  ctx_mask   [N, L] bool        valid-token mask
  null_ctx   [L, 512] fp16      empty-string embedding for CFG
  null_mask  [L] bool
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

MAX_TOKENS = 32

PACK_CAPTIONS = {
    "kenney-tiny-dungeon": "tiny dungeon tile, pixel art",
    "kenney-tiny-town": "tiny town tile, pixel art",
    "kenney-tiny-battle": "tiny battle map tile, pixel art",
    "kenney-micro-roguelike": "micro roguelike sprite, 8-bit pixel art",
    "kenney-pixel-platformer": "platformer game sprite, pixel art",
    "kenney-pixel-shmup": "spaceship shooter sprite, pixel art",
    "kenney-modern-city": "modern city tile, pixel art",
    "kenney-1bit": "1-bit style sprite, pixel art",
    "kenney-rpg": "roguelike object tile, pixel art",
    "kenney-characters": "roguelike character, pixel art sprite",
    "kenney-caves": "cave dungeon tile, pixel art",
}

DCSS_TEMPLATES = {
    "dcss-mon": "{} monster, pixel art sprite",
    "dcss-item": "{} item, pixel art sprite",
    "dcss-dungeon": "{} dungeon tile, pixel art",
    "dcss-effect": "{} spell effect, pixel art",
    "dcss-player": "{} equipment, pixel art sprite",
    "dcss-misc": "{}, pixel art sprite",
}


def clean(name: str) -> str:
    words = re.sub(r"[_\-]+", " ", name).strip()
    words = re.sub(r"\d+$", "", words).strip()  # trailing frame/variant numbers
    return words.lower()


def caption_for(group: str, name: str, annotations: dict[str, str]) -> str:
    if group.startswith("emoji/"):
        hexcode = name.upper().replace("_", "-")
        ann = annotations.get(hexcode)
        return f"{ann} emoji" if ann else f"{group.removeprefix('emoji/')} emoji"
    g = group.removeprefix("sprite/")
    if g in DCSS_TEMPLATES:
        return DCSS_TEMPLATES[g].format(clean(name))
    if g in PACK_CAPTIONS:
        return PACK_CAPTIONS[g]
    if g == "dawnlike":
        sheet = clean(name.rsplit("_", 2)[0])  # "{sheet}_{row}_{col}"
        return f"{sheet} sprite, pixel art"
    return "pixel art sprite"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/combined64.npz")
    ap.add_argument("--openmoji-json", default="data/raw/openmoji.json")
    ap.add_argument("--out", default="data/text64.npz")
    args = ap.parse_args()

    meta = json.loads(Path(args.openmoji_json).read_text(encoding="utf-8"))
    annotations = {e["hexcode"].upper(): e["annotation"] for e in meta}

    d = np.load(args.data)
    gn = [str(g) for g in d["group_names"]]
    captions = [
        caption_for(gn[gid], str(name), annotations)
        for gid, name in zip(d["group_ids"], d["names"])
    ]
    for c in captions[:3] + captions[-3:]:
        print("e.g.", repr(c))

    from transformers import CLIPTextModel, CLIPTokenizerFast

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    all_ctx, all_mask = [], []
    with torch.no_grad():
        for i in range(0, len(captions) + 1, 512):  # +1 slot: empty string appended last
            batch = (captions + [""])[i : i + 512]
            if not batch:
                break
            t = tok(batch, padding="max_length", truncation=True,
                    max_length=MAX_TOKENS, return_tensors="pt").to(device)
            out = enc(input_ids=t.input_ids, attention_mask=t.attention_mask)
            all_ctx.append(out.last_hidden_state.half().cpu())
            all_mask.append(t.attention_mask.bool().cpu())
    ctx = torch.cat(all_ctx).numpy()
    mask = torch.cat(all_mask).numpy()

    np.savez(
        args.out,
        captions=np.array(captions),
        ctx=ctx[:-1],
        ctx_mask=mask[:-1],
        null_ctx=ctx[-1],
        null_mask=mask[-1],
    )
    print(f"saved ctx {ctx[:-1].shape} fp16 -> {args.out}")


if __name__ == "__main__":
    main()
