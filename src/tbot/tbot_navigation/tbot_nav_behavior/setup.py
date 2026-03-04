from setuptools import find_packages, setup

package_name = 'tbot_nav_behavior'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='trungnh',
    maintainer_email='trungnh.aitech@gmail.com',
    description='Vision-based safety behavior: clamps cmd_vel and cancels Nav2 goals using 3D detections (warning/stop distances, corridor).',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # 'perception_nav_node = tbot_nav_behavior.perception_nav_node:main',
            'vision_safety_clamp_node = tbot_nav_behavior.vision_safety_clamp_node:main',
        ],
    },
)
