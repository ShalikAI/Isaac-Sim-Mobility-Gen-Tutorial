#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np

def inspect_normals(normals_root: str, num_frames: int):
    """
    For each camera folder under normals_root, load the first num_frames .npy files
    and report dtype, shape, min, max, mean, std.
    """
    cam_dirs = sorted(glob.glob(os.path.join(normals_root, '*')))
    if not cam_dirs:
        print(f"No camera folders found in {normals_root}")
        return

    for cam in cam_dirs:
        print(f"\nCamera folder: {os.path.basename(cam)}")
        files = sorted(glob.glob(os.path.join(cam, '*.npy')))[:num_frames]
        if not files:
            print("  (no .npy files)")
            continue

        for fpath in files:
            arr = np.load(fpath)
            print(f"  File: {os.path.basename(fpath)}")
            print(f"    dtype: {arr.dtype}")
            print(f"    shape: {arr.shape}")
            print(f"    min/max: {arr.min():.4f} / {arr.max():.4f}")
            print(f"    mean/std: {arr.mean():.4f} / {arr.std():.4f}")
        # break out early if you only want the first camera
        # remove the next line to inspect all cameras
        # break

def main():
    parser = argparse.ArgumentParser(
        description="Inspect normals (.npy) data for each camera"
    )
    parser.add_argument(
        "normals_root",
        help="Path to the 'state/normals' directory of your replay (e.g. .../state/normals)"
    )
    parser.add_argument(
        "--num", "-n", type=int, default=3,
        help="Number of frames to inspect per camera (default: 3)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.normals_root):
        print(f"Error: {args.normals_root} is not a directory")
        return

    inspect_normals(args.normals_root, args.num)

if __name__ == "__main__":
    main()
