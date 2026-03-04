from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    map_path = LaunchConfiguration(
        'map',
        default='/workspaces/vision-assisted-nav-sim/map_assets/map_house.yaml'
    )
    default_world = PathJoinSubstitution([
        FindPackageShare('tbot_gazebo'),
        'worlds',
        'turtlebot3_house_actors.world',
    ])
    world = LaunchConfiguration('world', default=default_world)
    x_pose = LaunchConfiguration('x_pose', default='-6.5')
    y_pose = LaunchConfiguration('y_pose', default='-2.5')

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

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('tbot_nav'),
                'launch',
                'nav2.launch.py',
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'map': map_path,
        }.items(),
    )

    control_tbot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('tbot_gazebo'),
                'launch',
                'control_tbot.launch.py',
            ])
        ]),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    vision_safety_clamp_node = Node(
        package='tbot_nav_behavior',
        executable='vision_safety_clamp_node',
        name='vision_safety_clamp_node',
        parameters=[{'enable_debug': True, 'use_sim_time': use_sim_time}],
        output='screen',
    )

    yolo_inference_node = Node(
        package='yolo_inference_ros',
        executable='yolo_inference_node',
        name='yolo_inference_node',
        parameters=[{
            'enable_debug': True,
            'model': 'yolov8n.pt',
            'fuse_model': True,
            'enable_3d': True,
            'enable_tracker': True,
            'use_sim_time': use_sim_time,
        }],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock',
        ),
        DeclareLaunchArgument(
            'map',
            default_value='/workspaces/vision-assisted-nav-sim/map_assets/map_house.yaml',
            description='Full path to map file to load',
        ),
        DeclareLaunchArgument(
            'world',
            default_value=default_world,
            description='Full path to Gazebo world file to load',
        ),
        DeclareLaunchArgument('x_pose', default_value='-6.5', description='Spawn x position'),
        DeclareLaunchArgument('y_pose', default_value='-2.5', description='Spawn y position'),
        sim_launch,
        nav2_launch,
        control_tbot_launch,
        yolo_inference_node,
        vision_safety_clamp_node,
    ])
