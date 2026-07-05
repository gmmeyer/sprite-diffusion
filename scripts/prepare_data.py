"""Rasterize emoji PNGs into a packed npz for training.

Composites RGBA onto a fixed white background (see "Gotcha: transparency"
in the project notes), resizes to the target size, and stores group labels
from openmoji.json for later class conditioning.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

BG_COLOR = (255, 255, 255)


def load_composited(path: Path, size: int) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        return None
    alpha = np.asarray(img)[..., 3]
    if alpha.max() == 0:  # fully transparent
        return None
    bg = Image.new("RGB", img.size, BG_COLOR)
    bg.paste(img, mask=img.getchannel("A"))
    bg = bg.resize((size, size), Image.LANCZOS)
    return np.asarray(bg, dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--openmoji-dir", default="openmoji-72")
    ap.add_argument("--twemoji-dir", default="twemoji-main/assets/72x72")
    args = ap.parse_args()
    out = args.out or Path(f"data/emoji{args.size}.npz")

    # hexcode -> group, from OpenMoji metadata (covers most Twemoji codepoints too)
    meta = json.loads((args.raw / "openmoji.json").read_text(encoding="utf-8"))
    hex_to_group = {e["hexcode"].upper(): e["group"] for e in meta}

    sources = [
        ("openmoji", sorted((args.raw / args.openmoji_dir).glob("*.png"))),
        ("twemoji", sorted((args.raw / args.twemoji_dir).glob("*.png"))),
    ]

    images, groups, srcs, names = [], [], [], []
    for src_name, files in sources:
        skipped = 0
        for f in tqdm(files, desc=src_name):
            arr = load_composited(f, args.size)
            if arr is None:
                skipped += 1
                continue
            hexcode = f.stem.upper().replace("_", "-")
            images.append(arr)
            groups.append(hex_to_group.get(hexcode, "unknown"))
            srcs.append(src_name)
            names.append(f.stem)
        print(f"{src_name}: {len(files) - skipped} kept, {skipped} skipped")

    group_names = sorted(set(groups))
    group_ids = np.array([group_names.index(g) for g in groups], dtype=np.int64)
    images = np.stack(images)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        images=images,
        group_ids=group_ids,
        group_names=np.array(group_names),
        sources=np.array(srcs),
        names=np.array(names),
    )
    print(f"saved {images.shape} -> {out}")
    print(f"groups ({len(group_names)}): {group_names}")


if __name__ == "__main__":
    main()
