"""Merge packed npz datasets (same image size) into one, unioning class labels.

    uv run python scripts/merge_npz.py --inputs data/emoji64.npz data/sprites64.npz \
        --prefixes emoji sprite --out data/combined64.npz
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--prefixes", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert len(args.inputs) == len(args.prefixes)

    images, group_strs, sources, names = [], [], [], []
    for path, prefix in zip(args.inputs, args.prefixes):
        d = np.load(path)
        gn = [str(g) for g in d["group_names"]]
        images.append(d["images"])
        group_strs += [f"{prefix}/{gn[i]}" for i in d["group_ids"]]
        sources += [str(s) for s in d["sources"]]
        names += [str(n) for n in d["names"]]
        print(f"{path}: {len(d['images'])} images, {len(gn)} groups")

    group_names = sorted(set(group_strs))
    lookup = {g: i for i, g in enumerate(group_names)}
    group_ids = np.array([lookup[g] for g in group_strs], dtype=np.int64)
    images_arr = np.concatenate(images)
    np.savez_compressed(
        args.out,
        images=images_arr,
        group_ids=group_ids,
        group_names=np.array(group_names),
        sources=np.array(sources),
        names=np.array(names),
    )
    print(f"saved {images_arr.shape}, {len(group_names)} groups -> {args.out}")


if __name__ == "__main__":
    main()
