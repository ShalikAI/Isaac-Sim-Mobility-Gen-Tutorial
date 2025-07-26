#!/usr/bin/env python3
import rclpy
from urdf_parser_py.urdf import URDF

def main():
    rclpy.init()

    # Path to your H1 URDF (make sure this is correct)
    urdf_path = 'urdf/h1.urdf'

    # Read the file as bytes
    with open(urdf_path, 'rb') as f:
        xml_bytes = f.read()

    # Parse from bytes (avoids the encoding‐declaration issue)
    robot = URDF.from_xml_string(xml_bytes)

    # Filter out fixed joints
    joint_names = [j.name for j in robot.joints if j.type != 'fixed']
    if len(joint_names) != 19:
        print(f"Warning: expected 19 joints but found {len(joint_names)}")

    print("H1 actuated joints (array index → name):")
    for i, name in enumerate(joint_names):
        print(f"  #{i:2d}: {name}")

    rclpy.shutdown()

if __name__ == '__main__':
    main()

