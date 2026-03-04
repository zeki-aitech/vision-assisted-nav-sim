from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='-6.5')
    y_pose = LaunchConfiguration('y_pose', default='-2.5')
    default_world = PathJoinSubstitution([
        FindPackageShare('tbot_gazebo'),
        'worlds',
        'turtlebot3_house_actors.world',
    ])
    world = LaunchConfiguration('world', default=default_world)

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('tbot_gazebo'),
                'launch',
                'sim.launch.py',
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'world': world,
            'x_pose': x_pose,
            'y_pose': y_pose,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock',
        ),
        DeclareLaunchArgument(
            'world',
            default_value=default_world,
            description='Full path to Gazebo world file to load',
        ),
        DeclareLaunchArgument('x_pose', default_value='-6.5', description='Spawn x position'),
        DeclareLaunchArgument('y_pose', default_value='-2.5', description='Spawn y position'),
        sim_launch,
    ])
