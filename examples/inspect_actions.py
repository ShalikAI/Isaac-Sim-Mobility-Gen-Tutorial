#!/usr/bin/env python3
import os
import glob
import numpy as np

def inspect_actions(common_folder, frame=0):
    files = sorted(glob.glob(os.path.join(common_folder, '*.npy')))
    if frame >= len(files):
        raise IndexError(f"Only {len(files)} frames available, cannot inspect frame {frame}")
    
    state = np.load(files[frame], allow_pickle=True).item()

    if 'robot.action' not in state:
        print(f"No 'robot.action' found in frame {frame}")
        return

    action = state['robot.action']
    print(f"Frame {frame:02d}:")
    print(f"  robot.action (shape {action.shape}, dtype={action.dtype}):")
    for i, a in enumerate(action):
        print(f"    #{i}: {a:.6f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Inspect robot.action array in a specific frame.")
    parser.add_argument('common_folder', help='Path to the state/common folder')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to inspect')
    args = parser.parse_args()

    inspect_actions(os.path.expanduser(args.common_folder), args.frame)
