from launch import LaunchContext, LaunchDescription
from launch.substitutions import EnvironmentVariable, PathJoinSubstitution
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

    use_sim_time = LaunchConfiguration('use_sim_time')

    lc = LaunchContext()
    joy_type = EnvironmentVariable('CPR_JOY_TYPE', default_value='ps5')


    filepath_config_joy = PathJoinSubstitution(
        [FindPackageShare('tbot_control'), 'config', ('teleop_' + joy_type.perform(lc) + '.yaml')]
    )

    node_joy = Node(
        namespace='joy_teleop',
        package='joy',
        executable='joy_node',
        output='screen',
        name='joy_node',
        parameters=[
            filepath_config_joy,
            {'use_sim_time': use_sim_time}
        ]
    )

    node_teleop_twist_joy = Node(
        namespace='joy_teleop',
        package='teleop_twist_joy',
        executable='teleop_node',
        output='screen',
        name='teleop_twist_joy_node',
        parameters=[
            filepath_config_joy,
            {'use_sim_time': use_sim_time}
        ]
    )


    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_argument)
    ld.add_action(node_joy)
    ld.add_action(node_teleop_twist_joy)
    return ld