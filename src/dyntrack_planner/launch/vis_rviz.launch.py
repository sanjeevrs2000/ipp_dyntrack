from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_name = 'dyntrack_planner'
    pkg_share = get_package_share_directory(pkg_name)
    
    config_arg = DeclareLaunchArgument(
        'config',
        default_value='def_config.rviz',
        description='name of the RViz config file to use'
    )

    rviz_config = PathJoinSubstitution([
        pkg_share, 'config', LaunchConfiguration('config')
    ])

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    return LaunchDescription([config_arg, rviz])
