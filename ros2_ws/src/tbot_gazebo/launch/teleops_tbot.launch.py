from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    tbot_control_share = FindPackageShare('tbot_control')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation/Gazebo clock'
    )

    teleop_joy_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([tbot_control_share, 'launch', 'teleop_joy.launch.py'])
        ]),
        launch_arguments=[('use_sim_time', LaunchConfiguration('use_sim_time'))]
    )



    return LaunchDescription([
        declare_use_sim_time,
        teleop_joy_launch,
        teleop_base_launch,
    ])
