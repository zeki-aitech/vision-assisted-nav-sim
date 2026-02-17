from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    declare_use_sim_time_argument = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation/Gazebo clock'
    )
    
    filepath_config_twist_mux = PathJoinSubstitution(
        [FindPackageShare('zbot_stella_n2_control'), 'config', 'twist_mux.yaml']
    )

    use_sim_time = LaunchConfiguration('use_sim_time')

    node_twist_mux = Node(
        package='twist_mux',
        executable='twist_mux',
        output='screen',
        remappings={('/cmd_vel_out', '/tbot_base/cmd_vel')},
        parameters=[
            filepath_config_twist_mux,
            {'use_sim_time': use_sim_time}
        ]
    )

    ld = LaunchDescription()
    ld.add_action(node_twist_mux)
    ld.add_action(declare_use_sim_time_argument)
    return ld