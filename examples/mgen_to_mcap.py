#!/usr/bin/env python3
"""
Convert a single MobilityGen replay directory into a ROS 2 MCAP bag.
Produces:
  - /base_pose        (geometry_msgs/PoseStamped)
  - /tf               (tf2_msgs/TFMessage)
  - /tf_static        (tf2_msgs/TFMessage, identity)
  - /rgb/image_raw/<camera>        (sensor_msgs/Image, rgb8)
  - /segmentation/image/<camera>   (sensor_msgs/Image, mono8)
  - /depth/image_raw/<camera>      (sensor_msgs/Image, mono16)
  - /normals/image/<camera>        (sensor_msgs/Image, rgb32f)
All data is assumed exported at 1 Hz (customizable).
"""
import os
import glob
import argparse
import numpy as np
from PIL import Image

import rclpy
from rclpy.serialization import serialize_message
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata

from sensor_msgs.msg import Image as ImgMsg
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge


def make_image_msg(npdata, encoding, stamp):
    bridge = CvBridge()
    # Choose the correct cv_bridge conversion
    if encoding == 'rgb8':
        return bridge.cv2_to_imgmsg(npdata, encoding='rgb8')
    elif encoding == 'mono16':
        return bridge.cv2_to_imgmsg(npdata, encoding='mono16')
    elif encoding == 'rgb32f':
        return bridge.cv2_to_imgmsg(npdata, encoding='32FC3')
    else:
        return bridge.cv2_to_imgmsg(npdata, encoding=encoding)


def loop_mod(folders, topic_base, read_fn, encoding, ext, writer, ts, nanosec):
    """
    Publish each sub-folder (camera) under its own topic.
      - folders: list of folder paths
      - topic_base: e.g. '/rgb/image_raw'
      - read_fn: function(path) -> numpy array
      - encoding: image encoding string
      - ext: file extension to replace '.npy' in common step ('.jpg', '.png', '.npy')
      - writer: rosbag2 writer
      - ts, nanosec: timestamp
    """
    for folder in folders:
        cam = os.path.basename(folder)  # e.g. "robot.front_camera.left.rgb_image"
        fname = os.path.basename(common_path).replace('.npy', ext)
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            continue
        data = read_fn(path)
        img_msg = make_image_msg(data, encoding, ts)
        img_msg.header.stamp = ts
        img_msg.header.frame_id = 'base_link'
        topic = f"{topic_base}/{cam}"
        writer.write(topic, serialize_message(img_msg), nanosec)


def main():
    parser = argparse.ArgumentParser(description='Convert MobilityGen replay to MCAP bag')
    parser.add_argument(
        '--input', required=True, type=str,
        help='path to one recording under replays (e.g. ~/MobilityGenData/replays/2025-07-19T10:08:32.202022)'
    )
    parser.add_argument(
        '--output', required=True, type=str,
        help='output bag filename (must end in .mcap, e.g. ~/bags/my_run.mcap)'
    )
    parser.add_argument(
        '--hz', type=float, default=1.0,
        help='data rate in Hz (default: 1.0)'
    )
    args = parser.parse_args()

    # Expand '~' if present
    base = os.path.expanduser(args.input)
    outbag = os.path.expanduser(args.output)

    # Initialize ROS 2 (for message definitions)
    rclpy.init()

    # Set up rosbag2 writer for MCAP
    writer = SequentialWriter()
    storage_opts = StorageOptions(uri=outbag, storage_id='mcap')
    conv_opts = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    writer.open(storage_opts, conv_opts)

    # Declare all topics
    topics = [
        TopicMetadata(name='/base_pose', type='geometry_msgs/msg/PoseStamped', serialization_format='cdr'),
        TopicMetadata(name='/tf',        type='tf2_msgs/msg/TFMessage',   serialization_format='cdr'),
        TopicMetadata(name='/tf_static', type='tf2_msgs/msg/TFMessage',   serialization_format='cdr'),
    ]
    # We'll add image topics dynamically below once we know camera names
    for meta in topics:
        writer.create_topic(meta)

    # Gather file lists
    common_list     = sorted(glob.glob(os.path.join(base, 'state', 'common', '*.npy')))
    rgb_folders     = sorted(glob.glob(os.path.join(base, 'state', 'rgb', '*')))
    seg_folders     = sorted(glob.glob(os.path.join(base, 'state', 'segmentation', '*')))
    depth_folders   = sorted(glob.glob(os.path.join(base, 'state', 'depth', '*')))
    normals_folders = sorted(glob.glob(os.path.join(base, 'state', 'normals', '*')))

    # Create image topics for each camera folder
    for folder_list, topic_base in [
        (rgb_folders, '/rgb/image_raw'),
        (seg_folders, '/segmentation/image'),
        (depth_folders, '/depth/image_raw'),
        (normals_folders, '/normals/image'),
    ]:
        for folder in folder_list:
            cam = os.path.basename(folder)
            writer.create_topic(TopicMetadata(
                name=f"{topic_base}/{cam}",
                type='sensor_msgs/msg/Image',
                serialization_format='cdr'
            ))

    # Publish a one-time static TF (identity) at t=0
    static_tf = TransformStamped()
    static_tf.header.stamp = rclpy.time.Time().to_msg()
    static_tf.header.frame_id = 'world'
    static_tf.child_frame_id = 'base_link'
    # default transform is identity
    static_msg = TFMessage(transforms=[static_tf])
    writer.write('/tf_static', serialize_message(static_msg), rclpy.time.Time(seconds=0, nanoseconds=0).nanoseconds)

    # Time stepping
    hz = args.hz
    dt_ns = int(1e9 / hz)

    # Loop through each common npy (one per timestamp)
    for idx, common_path in enumerate(common_list):
        # Build timestamp
        secs = int(idx / hz)
        nsecs = int((idx / hz - secs) * 1e9)
        t = rclpy.time.Time(seconds=secs, nanoseconds=nsecs)
        ts = t.to_msg()
        nanosec = t.nanoseconds

        # -- 1) /base_pose
        state = np.load(common_path, allow_pickle=True).item()
        # expecting state['position']=[x,y,z], state['orientation']=[x,y,z,w]
        pos = state.get('position', [0,0,0])
        ori = state.get('orientation', [0,0,0,1])
        pose = PoseStamped()
        pose.header.stamp = ts
        pose.header.frame_id = 'world'
        pose.pose.position.x = pos[0]
        pose.pose.position.y = pos[1]
        pose.pose.position.z = pos[2]
        pose.pose.orientation.x = ori[0]
        pose.pose.orientation.y = ori[1]
        pose.pose.orientation.z = ori[2]
        pose.pose.orientation.w = ori[3]
        writer.write('/base_pose', serialize_message(pose), nanosec)

        # -- 2) /tf (dynamic)
        tf = TransformStamped()
        tf.header.stamp = ts
        tf.header.frame_id = 'world'
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x = pos[0]
        tf.transform.translation.y = pos[1]
        tf.transform.translation.z = pos[2]
        tf.transform.rotation.x = ori[0]
        tf.transform.rotation.y = ori[1]
        tf.transform.rotation.z = ori[2]
        tf.transform.rotation.w = ori[3]
        tf_msg = TFMessage(transforms=[tf])
        writer.write('/tf', serialize_message(tf_msg), nanosec)

        # -- 3) images
        # RGB
        loop_mod(rgb_folders, '/rgb/image_raw',
                 lambda p: np.asarray(Image.open(p)), 'rgb8', '.jpg',
                 writer, ts, nanosec)

        # Segmentation (both instance_id & semantic are in same dir list)
        loop_mod(seg_folders, '/segmentation/image',
                 lambda p: np.asarray(Image.open(p)), 'mono8', '.png',
                 writer, ts, nanosec)

        # Depth (I;16 → mono16)
        loop_mod(depth_folders, '/depth/image_raw',
                 lambda p: np.asarray(Image.open(p).convert('I;16')), 'mono16', '.png',
                 writer, ts, nanosec)

        # Normals (npy → float32 RGB triplet)
        loop_mod(normals_folders, '/normals/image',
                 lambda p: np.load(p).astype(np.float32), 'rgb32f', '.npy',
                 writer, ts, nanosec)

    print(f"Finished writing MCAP bag: {outbag}")
    rclpy.shutdown()


if __name__ == '__main__':
    main()
