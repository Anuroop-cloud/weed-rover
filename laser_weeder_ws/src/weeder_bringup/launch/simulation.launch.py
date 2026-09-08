import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    # Package directories
    gazebo_pkg = get_package_share_directory('gazebo_ros')
    weeder_gazebo_pkg = get_package_share_directory('weeder_gazebo')
    weeder_desc_pkg = get_package_share_directory('weeder_description')
    weeder_bringup_pkg = get_package_share_directory('weeder_bringup')
    
    # Process URDF
    xacro_file = os.path.join(weeder_desc_pkg, 'urdf', 'weeder.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file)
    robot_description = {'robot_description': robot_description_config.toxml()}
    
    # Gazebo launch
    world_file = os.path.join(weeder_gazebo_pkg, 'worlds', 'farm.world')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_file}.items()
    )
    
    # Spawn robot
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'weeder', '-z', '0.1'],
        output='screen'
    )
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )
    
    # RViz2
    rviz_config_file = os.path.join(weeder_bringup_pkg, 'config', 'weeder.rviz')
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )
    
    # Spawn Field script (using ExecuteProcess to just run the python script)
    spawn_field_script = os.path.join(weeder_gazebo_pkg, 'scripts', 'spawn_field.py')
    spawn_field = ExecuteProcess(
        cmd=['python3', spawn_field_script],
        output='screen'
    )
    
    # Perception Node
    perception_node = Node(
        package='weeder_perception',
        executable='perception_node',
        output='screen'
    )
    
    # Control Node
    control_node = Node(
        package='weeder_control',
        executable='weed_eliminator_node',
        output='screen'
    )
    
    # Serial Bridge Node
    serial_node = Node(
        package='weeder_serial_bridge',
        executable='serial_bridge_node',
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_robot,
        spawn_field,
        perception_node,
        control_node,
        serial_node,
        rviz2
    ])
