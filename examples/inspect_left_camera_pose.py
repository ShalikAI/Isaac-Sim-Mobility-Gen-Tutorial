#!/usr/bin/env python3
import os, sys, glob
import numpy as np
from scipy.spatial.transform import Rotation as R

def inspect(folder, num=5):
    files = sorted(glob.glob(os.path.join(folder, '*.npy')))
    for i, fn in enumerate(files[:num]):
        st = np.load(fn, allow_pickle=True).item()

        # --- robot pelvis pose ---
        pos = st['robot.position']    # [x,y,z]
        ori = st['robot.orientation'] # [qx, qy, qz, qw]
        # quaternion → roll,pitch,yaw (XYZ order)
        yaw, pitch, roll = R.from_quat(ori).as_euler('xyz', degrees=True)

        print(f"Frame {i:02d}:")
        print(f"  pelvis pos  = {pos}")
        print(f"  pelvis quat = {ori}")
        print(f"    → yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°")

        # --- front‑left camera pose ---
        lpos = st['robot.front_camera.left.position']
        lori = st['robot.front_camera.left.orientation']
        lyaw, lroll, lpitch = R.from_quat(lori).as_euler('xyz', degrees=True)

        print(f"  left_cam pos  = {lpos}")
        print(f"  left_cam quat = {lori}")
        print(f"    → yaw={lyaw:.1f}°, pitch={lpitch:.1f}°, roll={lroll:.1f}°\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: inspect_poses.py /path/to/state/common [num]")
        sys.exit(1)
    folder = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
    inspect(folder, n)
