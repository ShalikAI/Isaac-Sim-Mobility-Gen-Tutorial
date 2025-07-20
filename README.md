<h1 align="center"><span> Isaac Sim Mobility Generator</span></h1>

<p align="center">
  <a href="https://www.youtube.com/watch?v=jR9Ikk9bB9w" target="_blank">
    <img src="assets/isaac_sim_mobility_gen.gif" alt="Video Thumbnail" width="80%">
  </a>
</p>


## Overview

MobilityGen is a toolset built on [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim) that enables you to easily generate and collect data for mobile robots.

It supports

- ***Rich ground truth data***

    https://github.com/user-attachments/assets/5e2c87c0-4255-4cdf-812e-104a43dd2f3c

    - Occupancy Map
    - Pose
    - Joint Positions / Velocities
    - RGB Images
    - Segmentation Images
    - Depth Images
    - Instance Segmentation Images
    - Normals Images

- ***Many robot types***

    - Differential drive - Jetbot, Carter
    - Quadruped - Spot
    - Humanoid - H1
    - *Implement your own by subclassing the [Robot](./exts/omni.ext.mobility_gen/omni/ext/mobility_gen/robots.py) class*

- ***Many data collection methods***

    - Manual - Keyboard Teleoperation, Gamepad Teleoperation
    - Automated - Random Accelerations, Random Path Following
    - *Implement your own by subclassing the [Scenario](./exts/omni.ext.mobility_gen/omni/ext/mobility_gen/scenarios.py) class*

This enables you to train models and test algorithms related to robot mobility.

To get started with MobilityGen follow the setup and usage instructions below!

## Table of Contents

- [🛠️ Setup](#setup)
- [👍 Basic Usage](#usage)
- [💡 How To Guides](#guides)
    - [How to record procedural data](#how-to-procedural-data)
    - [How to implement a custom robot](#how-to-custom-robot)
    - [How to implement a custom scenario](#how-to-custom-scenario)
- [📝 Data Format](#-data-format)
- [👏 Contributing](#-contributing)

<a id="setup"></a>
## 🛠️ Setup

Follow these steps to set up Isaac Sim Mobility Generator.

### Step 1 - Install Isaac Sim

1. Download [Isaac Sim 4.5.0](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html). Isaac Sim should be installed in ``~/isaacsim``.

### Step 2 - Clone this repository

1. Clone the repository

    ```bash
    https://github.com/ShalikAI/Isaac-Sim-Mobility-Gen-Tutorial.git
    ```

### Step 3 - Link Isaac Sim

Next, we'll call ``link_app.sh`` to link the Isaac Sim installation directory to the local ``app`` folder.

1. Navigate to the repo root

    ```bash
    cd Isaac-Sim-Mobility-Gen-Tutorial
    ```

2. Run the following to link the ``app`` folder and pass it the path to where you installed Isaac Sim

    ```bash
    ./link_app.sh --path ~/isaacsim
    ```

<details>
> This step is helpful as it (1) Enables us to use VS code autocompletion (2) Allows us to call ./app/python.sh to launch Isaac Sim Python scripts (3) Allows us to call ./app/isaac-sim.sh to launch Isaac Sim.
</details>

### Step 4 - Install other python dependencies (including C++ path planner) (for procedural generation)

1. Install miscellaneous python dependencies

    ```bash
    ./app/python.sh -m pip install tqdm
    ```

2. Navigate to the path planner directory

    ```bash
    cd Isaac-Sim-Mobility-Gen-Tutorial/path_planner
    ```

3. Install with pip using the Isaac Sim python interpreter

    ```bash
    ../app/python.sh -m pip install -e .
    ```

    > Note: If you run into an error related to pybind11 while running this command, you may try ``../app/python.sh -m pip install wheel`` and/or ``../app/python.sh -m pip install pybind11[global]``.

### Step 4 - Launch Isaac Sim

1. Navigate to the repo root

    ```bash
    cd MobilityGen
    ```

2. Launch Isaac Sim with required extensions enabled by calling

    ```bash
    ./scripts/launch_sim.sh
    ```

That's it!  If everything worked, you should see Isaac Sim open with a window titled ``MobilityGen`` appear.

<img src="./assets/extension_gui.png" height="640px">

Read [Usage](#usage) below to learn how to generate data with MobilityGen.

<a id="usage"></a>
## 👍 Basic Usage

Below details a typical workflow for collecting data with Isaac-Sim-Mobility-Gen-Tutorial.

### Step 1 - Launch Isaac Sim

1. Navigate to the repo root

    ```bash
    cd Isaac-Sim-Mobility-Gen-Tutorial
    ```

2. Launch Isaac Sim with required extensions enabled by calling

    ```bash
    ./scripts/launch_sim.sh
    ```

### Step 2 - Build a scenario

This assumes you see the MobilityGen extension window.

1. Under Scene USD URL / Path copy and paste the following

    ```
    http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd
    ```

2. Under the ``Scenario`` dropdown select ``KeyboardTeleoperationScenario`` to start

3. Under the ``Robot`` dropdown select ``H1Robot``

4. Click ``Build``

After a few seconds, you should see the scene and occupancy map appear.

### Step 3 - Initialize / reset the scenario

1. Click the ``Reset`` function to randomly initialize the scenario.  Do this until the robot spawns inside the warehouse.


### Step 4 - Test drive the robot

Before you start recording, try moving the robot around to get a feel for it

To move the robot, use the following keys

- ``W`` - Move Forward
- ``A`` - Turn Left
- ``S`` - Move Backwards
- ``D`` - Turn right

### Step 5 - Start recording!

Once you're comfortable, you can record a log.

1. Click ``Start Recording`` to start recording a log.

    > You should now see a recording name and the recording duration change.
2. Move the robot around
3. Click ``Stop Recording`` to stop recording.

The data is recorded to ``~/MobilityGenData/recordings`` by default.

### Step 6 - Render data

If you've gotten this far, you've recorded a trajectory, but it doesn't include the rendered sensor data.

Rendering the sensor data is done offline.  To do this call the following

1. Close Isaac Sim if it's running

2. Navigate to the repo root

    ```bash
    cd Isaac-Sim-Mobility-Gen-Tutorial
    ```

3. Run the ``scripts/replay_directory.py`` script to replay and render all recordings in the directory

    ```bash
    python scripts/replay_directory.py --render_interval=200
    ```

    > Note: For speed for this tutorial, we use a render interval of 200.  If our physics timestep is 200 FPS, this means we
    > render 1 image per second.

That's it! Now the data with renderings should be stored in ``~/MobilityGenData/replays``.

### Step 7 - Visualize the Data

We provide a few examples in the [examples](./examples) folder for working with the data.

One example is using Gradio to explore all of the recordings in the replays directory.  To run this example,
call the following

1. Install Gradio in your system:
    ```bash
    python3 -m pip install gradio
    ```
2. Call the gradio data visualization example script

    ```bash
    python examples/04_visualize_gradio.py
    ```

2. Open your web browser to ``http://127.0.0.1:7860`` to explore the data

If everything worked, you should be able to view the data in the browser.

<img src="assets/gradio_gui.png" height=320 />

### Next steps

That's it!  Once you've gotten the hang of how to record data, you might try

1. Record data using one of the procedural methods (like ``RandomAccelerationScenario`` or ``RandomPathFollowingScenario``).

    > These methods don't rely on human input, and automatically "restart" when finished to create new recordings.

2. Implement or customize your own [Robot](./exts/omni.ext.mobility_gen/omni/ext/mobility_gen/robots.py) class.
3. Implement or customize your own [Scenario](./exts/omni.ext.mobility_gen/omni/ext/mobility_gen/scenarios.py) class.

If you find MobilityGen helpful for your use case, run in to issues, or have any questions please [let us know!](https://github.com/NVlabs/MobilityGen/issues).

<a id="contributing"></a>

<a id="usage"></a>
## 💡 How To Guides

<a id="how-to-procedural-data"></a>
### How to record procedural data

#### Step 1 - Launch Isaac Sim

This is the same as in the basic usage.

1. Navigate to the repo root

    ```bash
    cd Isaac-Sim-Mobility-Gen-Tutorial
    ```

2. Launch Isaac Sim with required extensions enabled by calling

    ```bash
    ./scripts/launch_sim.sh
    ```

#### Step 2 - Build a scenario

1. Under Scene USD URL / Path copy and paste the following

    ```
    http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd
    ```

2. Under the ``Scenario`` dropdown select ``RandomPathFollowingScenario`` or ``RandomAccelerationScenario``

3. Under the ``Robot`` dropdown select ``H1Robot``

4. Click ``Build``

After a few seconds, you should see the scene and occupancy map appear.

#### Step 3 - Record data

1. Click ``Start Recording`` to start recording data

2. Go grab some coffee!

    > The procedural generated methods automatically determine when to reset (ie: if the robot collides with
    > an object and needs to respawn).  If you run into any issues with the procedural methods getting stuck, please let us know.

3. Click ``Stop Recording`` to stop recording data.

The data is recorded to ``~/MobilityGenData/recordings`` by default.

#### Step 4 - Render and visualize sensor data

This is the same as before.  Please refer to Step 6-7 of the "Basic Usage" guide.

<a id="how-to-custom-robot"></a>
### How to implement a new robot

You can implement a new robot for use with MobilityGen.

The general workflow is as follows:

1. Subclass the [Robot](exts/omni.ext.mobility_gen/omni/ext/mobility_gen/robots.py) class.
2. Implement the ``build()`` method.  This method is responsible for adding the robot to the USD stage.
2. Implement the ``write_action()`` method.  This method performs the logic of applying the linear, angular velocity command.
3. Overwrite the common class parameters (like ``physics_dt``, ``occupancy_map_z_min``, etc.)
4. Register the robot class by using the ``ROBOT.register()`` decorator.  This makes the custom robot discoverable.

We recommend referencing the example robots in [robots.py](exts/omni.ext.mobility_gen/omni/ext/mobility_gen/robots.py) for more details.

A good way to start could be simply by modifying an existing robot.  For example, you might change the position at which
the camera is mounted on the H1 robot.

<a id="how-to-custom-scenario"></a>
### How to implement a new scenario

You can implement a new data recording scenario for use with MobilityGen.

The general workflow is as follows:

1. Subclass the [Scenario](exts/omni.ext.mobility_gen/omni/ext/mobility_gen/scenarios.py) class.
2. Implement the ``reset()`` method.  This method is responsible for randomizing / initializing the scenario (ie: spawning the robot).
3. Implement the ``step()`` method.  This method is responsible for incrementing the scenario by one physics step.
4. Register the scenario class by using the ``SCENARIOS.register()`` decorator.  This makes the custom scenario discoverable.

We recommend referencing the example scenarios in [scenarios.py](exts/omni.ext.mobility_gen/omni/ext/mobility_gen/scenarios.py) for more details.

A good way to start could be simply by modifying an existing scenario.  For example, you might implement a new method
for generating random motions.

## Publish as ROS2 topics
You can publish the data once the rendering is finished. Convert the data as ROS2 topics:
```
python3 examples/mgen_to_mcap.py   --input ~/MobilityGenData/replays/2025-07-19T10:08:32.202022   --output ~/MobilityGenData/rosbags/2025-07-19.mcap   --hz 1.0
```
Here are the topics the rosbag will have:
```
arghya@arghya-Pulse-GL66-12UEK:~/MobilityGenData/rosbags$ ros2 bag info 2025-07-19.mcap/

Files:             2025-07-19.db3_0.db3
Bag size:          1.4 GiB
Storage id:        sqlite3
Duration:          67.000000000s
Start:             Dec 31 1969 19:00:00.000000000 (0.000000000)
End:               Dec 31 1969 19:01:07.000000000 (67.000000000)
Messages:          953
Topic information: Topic: /normals/image/robot_front_camera_left_normals_image | Type: sensor_msgs/msg/Image | Count: 68 | Serialization Format: cdr
                   Topic: /base_pose | Type: geometry_msgs/msg/PoseStamped | Count: 68 | Serialization Format: cdr
                   Topic: /tf_static | Type: tf2_msgs/msg/TFMessage | Count: 1 | Serialization Format: cdr
                   Topic: /tf | Type: tf2_msgs/msg/TFMessage | Count: 68 | Serialization Format: cdr
                   Topic: /segmentation/image/robot_front_camera_left_instance_id_segmentation_image | Type: sensor_msgs/msg/Image | Count: 68 | Serialization Format: cdr
                   Topic: /normals/image/robot_front_camera_right_normals_image | Type: sensor_msgs/msg/Image | Count: 68 | Serialization Format: cdr
                   Topic: /left_camera_pose | Type: geometry_msgs/msg/PoseStamped | Count: 68 | Serialization Format: cdr
                   Topic: /rgb/image_raw/robot_front_camera_left_rgb_image | Type: sensor_msgs/msg/Image | Count: 68 | Serialization Format: cdr
                   Topic: /joint_states | Type: sensor_msgs/msg/JointState | Count: 68 | Serialization Format: cdr
                   Topic: /depth/image_raw/robot_front_camera_right_depth_image | Type: sensor_msgs/msg/Image | Count: 68 | Serialization Format: cdr
                   Topic: /segmentation/image/robot_front_camera_left_segmentation_image | Type: sensor_msgs/msg/Image | Count: 68 | Serialization Format: cdr
                   Topic: /rgb/image_raw/robot_front_camera_right_rgb_image | Type: sensor_msgs/msg/Image | Count: 68 | Serialization Format: cdr
                   Topic: /segmentation/image/robot_front_camera_right_instance_id_segmentation_image | Type: sensor_msgs/msg/Image | Count: 68 | Serialization Format: cdr
                   Topic: /segmentation/image/robot_front_camera_right_segmentation_image | Type: sensor_msgs/msg/Image | Count: 68 | Serialization Format: cdr
                   Topic: /depth/image_raw/robot_front_camera_left_depth_image | Type: sensor_msgs/msg/Image | Count: 68 | Serialization Format: cdr
```
If you want to publish the robot state and visualize the robot model in rviz, do the following
```
ros2 run robot_state_publisher robot_state_publisher   --ros-args -p robot_description:="$(cat ~/Isaac-Sim-MobilityGen/h1_description/urdf/h1.urdf)"
```
Now, open the rviz and load the rviz file from this repo. You will see the output something like this below.

<p align="center">
  <a href="https://www.youtube.com/watch?v=jR9Ikk9bB9w" target="_blank">
    <img src="assets/isaac_sim_rosbag.gif" alt="Video Thumbnail" width="80%">
  </a>
</p>

## 📝 Data Format

MobilityGen records two types of data.

- *Static Data* is recorded at the beginning of a recording
    - Occupancy map
    - Configuration info
        - Robot type
        - Scenario type
        - Scene USD URL
    - USD Stage
- *State Data* is recorded at each physics timestep
    - Robot action: Linear, angular velocity
    - Robot pose: Position, quaternion
    - Robot joint Positions / Velocities
    - Robot sensor data:
        - Depth image
        - RGB Image
        - Segmentation image / info
        - Instance Segmentation
        - Normals

If you want to see the joints:
```
$ python3 examples/list_h1_joints.py 
```
Here is the output:
```
H1 actuated joints (array index → name):
  # 0: left_hip_yaw_joint
  # 1: left_hip_roll_joint
  # 2: left_hip_pitch_joint
  # 3: left_knee_joint
  # 4: left_ankle_joint
  # 5: right_hip_yaw_joint
  # 6: right_hip_roll_joint
  # 7: right_hip_pitch_joint
  # 8: right_knee_joint
  # 9: right_ankle_joint
  #10: torso_joint
  #11: left_shoulder_pitch_joint
  #12: left_shoulder_roll_joint
  #13: left_shoulder_yaw_joint
  #14: left_elbow_joint
  #15: right_shoulder_pitch_joint
  #16: right_shoulder_roll_joint
  #17: right_shoulder_yaw_joint
  #18: right_elbow_joint
```
If you want to see the keys:
```
python3 examples/inspect_common_folder.py ~/MobilityGenData/replays/2025-07-19T10:08:32.202022/state/common --num 1
```
The output was the following:
```
Found 68 .npy files. Inspecting first 1:

File: /home/arghya/MobilityGenData/replays/2025-07-19T10:08:32.202022/state/common/00000000.npy
  Type: <class 'numpy.ndarray'>
  Structure (keys and types/shapes):
    - robot.action: array, dtype=float64, shape=(2,)
    - robot.position: array, dtype=float32, shape=(3,)
    - robot.orientation: array, dtype=float32, shape=(4,)
    - robot.joint_positions: array, dtype=float32, shape=(19,)
    - robot.joint_velocities: array, dtype=float32, shape=(19,)
    - robot.front_camera.left.segmentation_info: <class 'dict'>, value={'idToLabels': {'0': {'class': 'BACKGROUND'}, '1': {'class': 'UNLABELLED'}, '2': {'class': 'rack'}, '3': {'class': 'pallet'}, '4': {'class': 'floor'}, '5': {'class': 'wall'}, '7': {'class': 'box'}, '8': {'class': 'pillar'}, '9': {'class': 'sign'}, '11': {'class': 'fire_extinguisher'}, '12': {'class': 'floor_decal'}, '13': {'class': 'crate'}}}
    - robot.front_camera.left.instance_id_segmentation_info: <class 'dict'>, value={'idToLabels': {'1': 'INVALID', '8': 'INVALID', '1892': '/World/robot/left_ankle_link/visuals', '1255': '/World/scene/PalletBin_02/Roller/SmallKLT_Visual_118/Visuals/FOF_Mesh_Label_1', '1686': '/World/scene/PalletBin_01/Roller/SmallKLT_Visual_130/Visuals/FOF_Mesh_Magenta_Box', '1601': '/World/scene/PalletBin_01/Roller/SmallKLT_Visual_132/Visuals/FOF_Mesh_Magenta_Box',
    ....
    '1349': '/World/scene/Shelf_0/S_AisleSign_47/S_AisleSign'}}
    - robot.front_camera.right.position: array, dtype=float32, shape=(3,)
    - robot.front_camera.right.orientation: array, dtype=float32, shape=(4,)
    - keyboard.buttons: array, dtype=bool, shape=(4,)
```
If you want to inspect joints:
```
python3 examples/inspect_joints.py ~/MobilityGenData/replays/2025-07-19T10:08:32.202022/state/common --frame 0
```
Here is the output:
```
Frame 00:
  Joint positions (shape (19,)):
    [ 0.0285178  -0.06889407 -0.00940297 -0.03465765 -0.02505827  0.3011327
  0.27076805 -0.6683024  -0.59880084 -0.0539261  -0.02191565  1.4317086
  1.2333937   0.0024948  -0.00682147 -0.6336425  -0.7075511   0.5251975
  0.5055799 ]
  Joint velocities(shape (19,)):
    [ 2.9184083e-02  1.4257843e-02 -2.9104811e-03  7.8836689e-03
 -3.1841312e-02 -2.3049731e-03  3.5154899e-03  5.5881063e-03
  5.8478154e-03 -6.4434828e-03  2.9966049e-03 -2.0087871e-03
  7.8600556e-02  8.9578883e-04  1.2077199e-03 -1.3268487e-04
 -1.3865213e+00 -1.3560086e-05  7.1849755e-04]

Index → value (position, velocity):
  # 0:  pos=0.0285, vel=0.0292
  # 1:  pos=-0.0689, vel=0.0143
  # 2:  pos=-0.0094, vel=-0.0029
  # 3:  pos=-0.0347, vel=0.0079
  # 4:  pos=-0.0251, vel=-0.0318
  # 5:  pos=0.3011, vel=-0.0023
  # 6:  pos=0.2708, vel=0.0035
  # 7:  pos=-0.6683, vel=0.0056
  # 8:  pos=-0.5988, vel=0.0058
  # 9:  pos=-0.0539, vel=-0.0064
  #10:  pos=-0.0219, vel=0.0030
  #11:  pos=1.4317, vel=-0.0020
  #12:  pos=1.2334, vel=0.0786
  #13:  pos=0.0025, vel=0.0009
  #14:  pos=-0.0068, vel=0.0012
  #15:  pos=-0.6336, vel=-0.0001
  #16:  pos=-0.7076, vel=-1.3865
  #17:  pos=0.5252, vel=-0.0000
  #18:  pos=0.5056, vel=0.0007
```

This data can easily be read using the [Reader](./examples/reader.py) class.

```python
from reader import Reader

reader = Reader(recording_path="replays/2025-01-17T16:44:33.006521")

print(len(reader)) # print number of timesteps

state_dict = reader.read_state_dict(0)  # read timestep 0
```

The state_dict has the following schema

```
{
    "robot.action": np.ndarray,                                      # [2] - Linear, angular command velocity
    "robot.position": np.ndarray,                                    # [3] - XYZ
    "robot.orientation": np.ndarray,                                 # [4] - Quaternion
    "robot.joint_positions": np.ndarray,                             # [J] - Joint positions
    "robot.joint_velocities": np.ndarray,                            # [J] - Joint velocities
    "robot.front_camera.left.rgb_image": np.ndarray,                 # [HxWx3], np.uint8 - RGB image
    "robot.front_camera.left.depth_image": np.ndarray,               # [HxW], np.fp32 - Depth in meters
    "robot.front_camera.left.segmentation_image": np.ndarray,        # [HxW], np.uint8 - Segmentation class index
    "robot.front_camera.left.segmentation_info": dict,               # see Isaac replicator segmentation info format
    "robot.front_camera.left.position": np.ndarray,                  # [3] - XYZ camera world position
    "robot.front_camera.left.orientation": np.ndarray,               # [4] - Quaternion camera world orientation
    ...
}
```

The ``Reader`` class abstracts away the details of reading the state dictionary
from the recording.

In case you're interested, each recording is represented as a directory with the following structure

```
2025-01-17T16:44:33.006521/
    occupancy_map/
        map.png
        map.yaml
    config.json
    stage.usd
    state/
        common/
            00000000.npy
            00000001.npy
            ...
        depth/
            robot.front_camera.left.depth_image/
                00000000.png
                00000001.png
                ...
            robot.front_camera.right.depth_image/
                ...
        rgb/
            robot.front_camera.left.rgb_image/
                00000000.jpg
                00000001.jpg
            robot.front_camera.right.rgb_image/
                ...
        segmentation/
            robot.front_camera.left.segmentation_image/
                00000000.png
                00000001.png
                ...
            robot.front_camera.right.segmentation_image/
                ...
        normals/
            robot.front_camera.left.normals_image\
                00000000.npy
                00000001.npy
                ...

```

Most of the state information is captured under the ``state/common`` folder, as dictionary in a single ``.npy`` file.

However, for some data (images) this is inefficient.  These instead get captured in their own folder based on the data
type and the name.  (ie: rgb/robot.front_camera.left.depth_image).

The name of each file corresponds to its physics timestep.

If you have any questions regarding the data logged by MobilityGen, please [let us know!](https://github.com/NVlabs/MobilityGen/issues)

### Converting to the LeRobot Format

A script that converts a MobilityGen recording/replay to a [LeRobot](https://github.com/huggingface/lerobot) dataset can be found at [scripts/convert_to_lerobot.py](./scripts/convert_to_lerobot.py). The [LeRobot Python package](https://github.com/huggingface/lerobot?tab=readme-ov-file#installation) needs to be installed before executing the script.

Example usage for converting a single recording:
```
python ./scripts/convert_to_lerobot.py \
  --input "~/MobilityGenData/replays/2025-00-00T00:00:00.000000" \
  --output "/output/path" \
  --fps 30
```

Example usage for converting a collection of recordings:
```
python ./scripts/convert_to_lerobot.py \
  --input "/path/to/directory/containing/the/recordings" \
  --output "/output/path" \
  --batch \
  --fps 30
```