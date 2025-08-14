from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os.path



# def generate_launch_description():

#     rvz = Node(
#             package='rviz2',
#             namespace='',
#             executable='rviz2',
#             name='rviz2',
#             arguments=['-d' + os.path.join(get_package_share_directory('vrx_gazebo'), 'config', 'rviz_vrx_rsp.rviz')]
#         )
#     return LaunchDescription([rvz])

def generate_launch_description():
    # 1) Declare a launch argument called "config_file"
    config_arg = DeclareLaunchArgument(
        'config',
        default_value='rviz_vrx_rsp.rviz',
        description='Name of the RViz config file in vrx_gazebo/config/'
    )

    # 2) Build the full path: <pkg_share>/config/<config_file>
    pkg_share = get_package_share_directory('vrx_gazebo')
    rviz_config = PathJoinSubstitution([
        pkg_share, 'config', LaunchConfiguration('config')
    ])

    # 3) Launch RViz with "-d <full config path>"
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config]
    )

    return LaunchDescription([
        config_arg,
        rviz_node
    ])