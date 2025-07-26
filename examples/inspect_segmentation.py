#!/usr/bin/env python3
import os
import glob
import argparse
from collections import Counter

import numpy as np
from PIL import Image

def inspect_folder(folder_path, num_frames=1):
    """
    Load up to `num_frames` files in folder_path (sorted), report:
      - dtype, shape
      - unique value count
      - top-5 most common labels
    """
    # collect all image files (png)
    files = sorted(glob.glob(os.path.join(folder_path, '*.png')))
    print(f"\nFolder: {os.path.basename(folder_path)}")
    print(f"  Total files: {len(files)}")
    for fn in files[:num_frames]:
        arr = np.asarray(Image.open(fn))
        flat = arr.flatten()
        uniq, counts = np.unique(flat, return_counts=True)
        cnt = dict(zip(uniq, counts))
        top5 = Counter(cnt).most_common(5)
        print(f"  File: {os.path.basename(fn)}")
        print(f"    dtype: {arr.dtype}, shape: {arr.shape}")
        print(f"    unique labels: {len(uniq)}")
        print(f"    top-5 labels (label:count): {top5}")

def main():
    p = argparse.ArgumentParser(description="Inspect segmentation folders")
    p.add_argument('seg_dir', help="path to state/segmentation")
    p.add_argument('--num', type=int, default=1,
                   help="how many files per folder to inspect")
    args = p.parse_args()

    seg_root = os.path.expanduser(args.seg_dir)
    subfolders = sorted(glob.glob(os.path.join(seg_root, '*')))
    if not subfolders:
        print(f"No subfolders found in {seg_root}")
        return

    for folder in subfolders:
        inspect_folder(folder, num_frames=args.num)

if __name__ == '__main__':
    main()
