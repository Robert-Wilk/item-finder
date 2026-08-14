"""
Brings up the three item_finder nodes.

Does NOT launch your existing base-robot nodes (camera_node,
ultrasonic_node, imu_node, base_driver, safety_node, teleop) - those
should already be running from your base robot's own launch file.
This file is additive: it assumes /image_raw, /cmd_vel_raw (consumed
by your safety_node), etc. already exist in the graph.

Run alongside your existing base launch, e.g.:
  ros2 launch <your_base_pkg> base.launch.py &
  ros2 launch item_finder item_finder.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    image_topic_arg = DeclareLaunchArgument(
        'image_topic', default_value='/image_raw',
        description='Camera topic for perception_node to subscribe to'
    )

    return LaunchDescription([
        image_topic_arg,

        Node(
            package='item_finder',
            executable='perception_node',
            name='perception_node',
            output='screen',
            parameters=[{
                'image_topic': LaunchConfiguration('image_topic'),
                'confidence_threshold': 0.45,
                'inference_hz': 2.0,
                'image_size': 320,
            }],
        ),

        Node(
            package='item_finder',
            executable='query_matcher_node',
            name='query_matcher_node',
            output='screen',
        ),

        Node(
            package='item_finder',
            executable='search_state_node',
            name='search_state_node',
            output='screen',
        ),
    ])
