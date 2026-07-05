"""Build a packed sprite npz from the downloaded packs (see docs/sprite-sources.md).

Handles two source kinds:
  files  — individual sprite PNGs (size-filtered so previews/sheets are excluded)
  sheet  — spritesheets sliced on a fixed grid (tile size + optional 1px spacing)

Cells/sprites are composited onto white, integer nearest-neighbor upscaled to
the target size, centered, deduped, and saved with pack-level class labels.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

BG_COLOR = (255, 255, 255)
MIN_ALPHA_COVERAGE = 0.05  # drop cells with <5% opaque pixels
SHEET_MIN_DIM = 200  # a PNG this large inside a sheet pack is a spritesheet

SOURCES = [
    {"kind": "files", "label": "dcss-mon", "root": "crawl/crawl-ref/source/rltiles/mon"},
    {"kind": "files", "label": "dcss-item", "root": "crawl/crawl-ref/source/rltiles/item"},
    {"kind": "files", "label": "dcss-player", "root": "crawl/crawl-ref/source/rltiles/player"},
    {"kind": "files", "label": "dcss-dungeon", "root": "crawl/crawl-ref/source/rltiles/dngn"},
    {"kind": "files", "label": "dcss-effect", "root": "crawl/crawl-ref/source/rltiles/effect"},
    {"kind": "files", "label": "dcss-misc", "root": "crawl/crawl-ref/source/rltiles/misc"},
    {"kind": "files", "label": "kenney-tiny-dungeon", "root": "kenney/tiny-dungeon"},
    {"kind": "files", "label": "kenney-tiny-town", "root": "kenney/tiny-town"},
    {"kind": "files", "label": "kenney-tiny-battle", "root": "kenney/tiny-battle"},
    {"kind": "files", "label": "kenney-micro-roguelike", "root": "kenney/micro-roguelike"},
    {"kind": "files", "label": "kenney-pixel-platformer", "root": "kenney/pixel-platformer"},
    {"kind": "files", "label": "kenney-pixel-shmup", "root": "kenney/pixel-shmup"},
    {"kind": "files", "label": "kenney-modern-city", "root": "kenney/roguelike-modern-city"},
    {"kind": "sheet", "label": "kenney-1bit", "root": "kenney/1-bit-pack",
     "match": "colored_packed", "tile": 16, "spacing": 0},
    {"kind": "sheet", "label": "kenney-rpg", "root": "kenney/roguelike-rpg-pack",
     "tile": 16, "spacing": 1},
    {"kind": "sheet", "label": "kenney-characters", "root": "kenney/roguelike-characters",
     "tile": 16, "spacing": 1},
    {"kind": "sheet", "label": "kenney-caves", "root": "kenney/roguelike-caves-dungeons",
     "tile": 16, "spacing": 1},
    {"kind": "sheet", "label": "dawnlike", "root": "dawnlike", "tile": 16, "spacing": 0,
     "exclude": ("GUI", "Examples", "Commissions")},
]


def render(cell: Image.Image, target: int) -> np.ndarray | None:
    """RGBA cell -> white-composited, integer-upscaled, centered target x target RGB."""
    arr = np.asarray(cell).copy()
    # magenta colorkey: some sheets mark transparency as (255,0,255) instead of alpha
    magenta = (arr[..., 0] == 255) & (arr[..., 1] == 0) & (arr[..., 2] == 255)
    if magenta.any():
        arr[..., 3][magenta] = 0
        cell = Image.fromarray(arr)
    alpha = arr[..., 3]
    if (alpha > 16).mean() < MIN_ALPHA_COVERAGE:
        return None
    k = max(1, target // max(cell.size))
    if k > 1:
        cell = cell.resize((cell.width * k, cell.height * k), Image.NEAREST)
    canvas = Image.new("RGB", (target, target), BG_COLOR)
    off = ((target - cell.width) // 2, (target - cell.height) // 2)
    if cell.width > target or cell.height > target:  # oversized odd tile: downscale
        cell.thumbnail((target, target), Image.LANCZOS)
        off = ((target - cell.width) // 2, (target - cell.height) // 2)
    canvas.paste(cell, off, mask=cell.getchannel("A"))
    return np.asarray(canvas, dtype=np.uint8)


def iter_files(src: dict, raw: Path, max_px: int = 40):
    for f in sorted((raw / src["root"]).rglob("*.png")):
        try:
            img = Image.open(f).convert("RGBA")
        except Exception:
            continue
        if max(img.size) > max_px:  # preview image or packed sheet
            continue
        yield f.stem, img


def iter_sheet_cells(src: dict, raw: Path):
    tile, sp = src["tile"], src["spacing"]
    for f in sorted((raw / src["root"]).rglob("*.png")):
        if any(part in src.get("exclude", ()) for part in f.parts):
            continue
        # packs ship Preview/Sample scene images that must not be sliced as sheets
        if f.stem.lower().startswith(("preview", "sample")):
            continue
        if src.get("match") and src["match"] not in f.stem:
            continue
        try:
            img = Image.open(f).convert("RGBA")
        except Exception:
            continue
        if not src.get("match") and max(img.size) < SHEET_MIN_DIM:
            continue
        cols = (img.width + sp) // (tile + sp)
        rows = (img.height + sp) // (tile + sp)
        for r in range(rows):
            for c in range(cols):
                x, y = c * (tile + sp), r * (tile + sp)
                yield f"{f.stem}_{r}_{c}", img.crop((x, y, x + tile, y + tile))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(f"data/sprites{args.size}.npz")

    images, groups, names = [], [], []
    seen: set[bytes] = set()
    for src in SOURCES:
        if not (args.raw / src["root"]).exists():
            print(f"MISSING: {src['root']} — skipped")
            continue
        cells = iter_files(src, args.raw) if src["kind"] == "files" else iter_sheet_cells(src, args.raw)
        kept = dropped = 0
        for name, cell in tqdm(cells, desc=src["label"], leave=False):
            arr = render(cell, args.size)
            if arr is None:
                dropped += 1
                continue
            key = arr.tobytes()
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            images.append(arr)
            groups.append(src["label"])
            names.append(name)
            kept += 1
        print(f"{src['label']}: {kept} kept, {dropped} dropped")

    group_names = sorted(set(groups))
    group_ids = np.array([group_names.index(g) for g in groups], dtype=np.int64)
    images_arr = np.stack(images)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        images=images_arr,
        group_ids=group_ids,
        group_names=np.array(group_names),
        sources=np.array(groups),
        names=np.array(names),
    )
    print(f"saved {images_arr.shape} -> {out}")
    print(f"groups ({len(group_names)}): {group_names}")


if __name__ == "__main__":
    main()
