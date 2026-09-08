import os

ws_dir = '/home/rithwik/Desktop/C2C/laser_weeder_ws/src'

def write_file(path, content):
    full_path = os.path.join(ws_dir, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w') as f:
        f.write(content.strip() + '\n')

# Weeder Description (ament_cmake)
write_file('weeder_description/package.xml', '''
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>weeder_description</name>
  <version>0.0.0</version>
  <description>URDF description for laser weeder</description>
  <maintainer email="user@todo.todo">user</maintainer>
  <license>TODO: License declaration</license>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>joint_state_publisher</exec_depend>
  <exec_depend>xacro</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
''')

write_file('weeder_description/CMakeLists.txt', '''
cmake_minimum_required(VERSION 3.8)
project(weeder_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf rviz meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
''')

# Weeder Gazebo (ament_cmake)
write_file('weeder_gazebo/package.xml', '''
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>weeder_gazebo</name>
  <version>0.0.0</version>
  <description>Gazebo simulation environment for laser weeder</description>
  <maintainer email="user@todo.todo">user</maintainer>
  <license>TODO: License declaration</license>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <exec_depend>gazebo_ros_pkgs</exec_depend>
  <exec_depend>weeder_description</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
    <gazebo_ros gazebo_model_path="${prefix}/models"/>
  </export>
</package>
''')

write_file('weeder_gazebo/CMakeLists.txt', '''
cmake_minimum_required(VERSION 3.8)
project(weeder_gazebo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch worlds models scripts
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
''')

# Weeder Bringup (ament_cmake)
write_file('weeder_bringup/package.xml', '''
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>weeder_bringup</name>
  <version>0.0.0</version>
  <description>Bringup for laser weeder</description>
  <maintainer email="user@todo.todo">user</maintainer>
  <license>TODO: License declaration</license>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <exec_depend>weeder_description</exec_depend>
  <exec_depend>weeder_gazebo</exec_depend>
  <exec_depend>weeder_perception</exec_depend>
  <exec_depend>weeder_control</exec_depend>
  <exec_depend>weeder_serial_bridge</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
''')

write_file('weeder_bringup/CMakeLists.txt', '''
cmake_minimum_required(VERSION 3.8)
project(weeder_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
''')

# Weeder Perception (ament_python)
write_file('weeder_perception/package.xml', '''
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>weeder_perception</name>
  <version>0.0.0</version>
  <description>Perception node for weed detection</description>
  <maintainer email="user@todo.todo">user</maintainer>
  <license>TODO: License declaration</license>

  <exec_depend>rclpy</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>cv_bridge</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
''')

write_file('weeder_perception/setup.py', '''
from setuptools import setup
import os
from glob import glob

package_name = 'weeder_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Perception node for weed detection',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = weeder_perception.perception_node:main'
        ],
    },
)
''')
write_file('weeder_perception/resource/weeder_perception', '')
write_file('weeder_perception/weeder_perception/__init__.py', '')

# Weeder Control (ament_python)
write_file('weeder_control/package.xml', '''
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>weeder_control</name>
  <version>0.0.0</version>
  <description>Control logic for weed elimination</description>
  <maintainer email="user@todo.todo">user</maintainer>
  <license>TODO: License declaration</license>

  <exec_depend>rclpy</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  <exec_depend>visualization_msgs</exec_depend>
  <exec_depend>gazebo_msgs</exec_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
''')

write_file('weeder_control/setup.py', '''
from setuptools import setup
import os
from glob import glob

package_name = 'weeder_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Control node for laser weeder',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'weed_eliminator_node = weeder_control.weed_eliminator_node:main'
        ],
    },
)
''')
write_file('weeder_control/resource/weeder_control', '')
write_file('weeder_control/weeder_control/__init__.py', '')

# Weeder Serial Bridge (ament_python)
write_file('weeder_serial_bridge/package.xml', '''
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>weeder_serial_bridge</name>
  <version>0.0.0</version>
  <description>Stub serial bridge for ESP32</description>
  <maintainer email="user@todo.todo">user</maintainer>
  <license>TODO: License declaration</license>

  <exec_depend>rclpy</exec_depend>
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
''')

write_file('weeder_serial_bridge/setup.py', '''
from setuptools import setup
import os
from glob import glob

package_name = 'weeder_serial_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Serial bridge for ESP32 communication',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'serial_bridge_node = weeder_serial_bridge.serial_bridge_node:main'
        ],
    },
)
''')
write_file('weeder_serial_bridge/resource/weeder_serial_bridge', '')
write_file('weeder_serial_bridge/weeder_serial_bridge/__init__.py', '')
