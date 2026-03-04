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
        default='/workspaces/vision-assisted-nav-sim/map_assets/map_house.yaml',
    )
    model_path = LaunchConfiguration(
        'model',
        default='/workspaces/vision-assisted-nav-sim/test_temp/yolov8n.engine',
    )
    tracker_cfg_path = LaunchConfiguration(
        'tracker_cfg',
        default=PathJoinSubstitution([
            FindPackageShare('yolo_inference_ros'),
            'config',
            'bytetrack.yaml',
        ]),
    )
    threshold = LaunchConfiguration('threshold', default='0.25')

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

    yolo_inference_node = Node(
        package='yolo_inference_ros',
        executable='yolo_inference_node',
        name='yolo_inference_node',
        parameters=[{
            'enable_debug': True,
            'model': model_path,
            'fuse_model': True,
            'enable_3d': True,
            'enable_tracker': True,
            'tracker_cfg': tracker_cfg_path,
            'threshold': threshold,
            'use_sim_time': use_sim_time,
        }],
        output='screen',
    )

    vision_safety_clamp_node = Node(
        package='tbot_nav_behavior',
        executable='vision_safety_clamp_node',
        name='vision_safety_clamp_node',
        parameters=[{'enable_debug': True, 'use_sim_time': use_sim_time}],
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
            default_value='',
            description='Full path to map file to load',
        ),
        DeclareLaunchArgument(
            'model',
            default_value='',
            description='Full path to YOLO model file (.pt or .engine)',
        ),
        DeclareLaunchArgument(
            'tracker_cfg',
            default_value=PathJoinSubstitution([
                FindPackageShare('yolo_inference_ros'),
                'config',
                'bytetrack.yaml',
            ]),
            description='Full path to ByteTrack config YAML',
        ),
        DeclareLaunchArgument(
            'threshold',
            default_value='0.25',
            description='YOLO detection confidence threshold',
        ),
        control_tbot_launch,
        nav2_launch,
        yolo_inference_node,
        vision_safety_clamp_node,
    ])
