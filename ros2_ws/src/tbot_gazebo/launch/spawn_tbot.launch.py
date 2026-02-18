
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import FindExecutable
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Get the urdf file
    # TBOT_MODEL = os.environ['TBOT_MODEL']
    TBOT_MODEL = 'wf'
    model_folder = 'tbot_' + TBOT_MODEL
    urdf_path = os.path.join(
        get_package_share_directory('tbot_gazebo'),
        'models',
        model_folder,
        'model.sdf'
    )
    
    # TBOT_MODEL = 'tbot'
    # urdf_path = os.path.join(
    #     get_package_share_directory('tbot_description'),
    #     'urdf',
    #     'tbot.urdf.xacro'
    # )
    
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    robot_desc = ParameterValue(
        Command([
            PathJoinSubstitution([FindExecutable(name='xacro')]),
            ' ',
            PathJoinSubstitution(
                [FindPackageShare('tbot_description'), 'urdf', 'tbot.urdf.xacro']
            ),
            ' ',
            'is_sim:=', use_sim_time
        ])
    )

    # Launch configuration variables specific to simulation
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')

    # Declare the launch arguments
    declare_x_position_cmd = DeclareLaunchArgument(
        'x_pose', default_value='0.0',
        description='Specify namespace of the robot')

    declare_y_position_cmd = DeclareLaunchArgument(
        'y_pose', default_value='0.0',
        description='Specify namespace of the robot')

    start_gazebo_ros_spawner_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', TBOT_MODEL,
            # '-file', urdf_path,
            '-topic', 'robot_description',
            '-x', x_pose,
            '-y', y_pose,
            '-z', '0.05',
            '-Y', '1.57'
        ],
        output='screen',
    )

    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_x_position_cmd)
    ld.add_action(declare_y_position_cmd)

    # Add any conditioned actions
    ld.add_action(start_gazebo_ros_spawner_cmd)

    return ld