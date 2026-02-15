import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('vision_assisted_nav_sim')
    default_model_path = os.path.join(pkg_share, 'models', 'human', 'model.sdf')

    declare_entity_name = DeclareLaunchArgument(
        'entity_name',
        default_value='human',
        description='Name of the spawned entity in Gazebo',
    )
    declare_x = DeclareLaunchArgument(
        'x',
        default_value='0.0',
        description='Initial x position (m)',
    )
    declare_y = DeclareLaunchArgument(
        'y',
        default_value='0.0',
        description='Initial y position (m)',
    )
    declare_z = DeclareLaunchArgument(
        'z',
        default_value='0.0',
        description='Initial z position (m)',
    )
    declare_yaw = DeclareLaunchArgument(
        'yaw',
        default_value='0.0',
        description='Initial yaw (radians)',
    )
    declare_model_file = DeclareLaunchArgument(
        'model_file',
        default_value=default_model_path,
        description='Path to human model SDF or URDF file',
    )
    declare_timeout = DeclareLaunchArgument(
        'timeout',
        default_value='30.0',
        description='Timeout in seconds for spawn service',
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_human',
        output='screen',
        arguments=[
            '-entity', LaunchConfiguration('entity_name'),
            '-file', LaunchConfiguration('model_file'),
            '-x', LaunchConfiguration('x'),
            '-y', LaunchConfiguration('y'),
            '-z', LaunchConfiguration('z'),
            '-Y', LaunchConfiguration('yaw'),
            '-timeout', LaunchConfiguration('timeout'),
        ],
    )

    ld = LaunchDescription([
        declare_entity_name,
        declare_x,
        declare_y,
        declare_z,
        declare_yaw,
        declare_model_file,
        declare_timeout,
        spawn_entity,
    ])

    return ld
