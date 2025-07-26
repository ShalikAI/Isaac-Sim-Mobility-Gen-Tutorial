import os
import glob
import argparse
import numpy as np
import pprint

def inspect_common_folder(common_folder, num_files=3):
    # Find all .npy files
    files = sorted(glob.glob(os.path.join(common_folder, '*.npy')))
    if not files:
        print(f"No .npy files found in {common_folder}")
        return

    print(f"Found {len(files)} .npy files. Inspecting first {min(num_files, len(files))}:\n")

    for file in files[:num_files]:
        print(f"File: {file}")
        data = np.load(file, allow_pickle=True)
        print(f"  Type: {type(data)}")
        # If it's a dict-like
        if isinstance(data, np.ndarray) and data.dtype == 'object':
            data = data.item()  # convert 0-d object array to dict
        if isinstance(data, dict):
            print("  Structure (keys and types/shapes):")
            for k, v in data.items():
                v_type = type(v)
                if isinstance(v, np.ndarray):
                    print(f"    - {k}: array, dtype={v.dtype}, shape={v.shape}")
                else:
                    print(f"    - {k}: {v_type}, value={v}")
        else:
            print("  Content:")
            pprint.pprint(data)
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect common folder .npy structure")
    parser.add_argument("common_folder", help="Path to the 'state/common' folder")
    parser.add_argument("--num", type=int, default=3, help="Number of files to inspect")
    args = parser.parse_args()

    inspect_common_folder(os.path.expanduser(args.common_folder), num_files=args.num)
