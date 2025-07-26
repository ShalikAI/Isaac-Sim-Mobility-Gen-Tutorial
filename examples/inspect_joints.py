#!/usr/bin/env python3
import os, glob
import numpy as np

def inspect_joints(common_folder, frame=0):
    files = sorted(glob.glob(os.path.join(common_folder, '*.npy')))
    if frame >= len(files):
        raise IndexError(f"Only {len(files)} frames, cannot inspect frame {frame}")
    state = np.load(files[frame], allow_pickle=True).item()

    jp = state['robot.joint_positions']
    jv = state['robot.joint_velocities']

    print(f"Frame {frame:02d}:")
    print(f"  Joint positions (shape {jp.shape}):\n    {jp}")
    print(f"  Joint velocities(shape {jv.shape}):\n    {jv}")
    print("\nIndex → value (position, velocity):")
    for i, (p, v) in enumerate(zip(jp, jv)):
        print(f"  #{i:2d}:  pos={p:.4f}, vel={v:.4f}")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('common_folder', help='state/common')
    p.add_argument('--frame', type=int, default=0, help='which frame index to inspect')
    args = p.parse_args()
    inspect_joints(os.path.expanduser(args.common_folder), args.frame)
