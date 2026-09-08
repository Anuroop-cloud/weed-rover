#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from std_msgs.msg import String
import os
import random
import json
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy


class FieldSpawner(Node):
    def __init__(self):
        super().__init__('field_spawner')
        self.cli = self.create_client(SpawnEntity, '/spawn_entity')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /spawn_entity service...')

        # Latched publisher so the eliminator can subscribe at any time
        latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.weed_pub = self.create_publisher(String, '/weeder/weed_map', latched_qos)

        self.spawn_field()

    def spawn_field(self):
        pkg_path = get_package_share_directory('weeder_gazebo')
        red_sdf_path  = os.path.join(pkg_path, 'models', 'red_weed',   'model.sdf')
        black_sdf_path = os.path.join(pkg_path, 'models', 'black_crop', 'model.sdf')

        try:
            with open(red_sdf_path, 'r') as f:
                red_sdf = f.read()
            with open(black_sdf_path, 'r') as f:
                black_sdf = f.read()
        except Exception as e:
            self.get_logger().error(f'Failed to read SDF files: {e}')
            return

        weed_idx = 0
        crop_idx = 0
        weed_map = {}   # name -> {x, y}

        # 4 rows of crops, 15 columns each
        num_rows = 4
        num_cols = 15
        row_spacing  = 0.45   # distance between crop rows
        col_spacing  = 0.18   # distance between crops along a row
        start_x = -1.0
        start_y = -0.7

        for row in range(num_rows):
            y_pos = start_y + row * row_spacing
            for col in range(num_cols):
                x_pos = start_x + col * col_spacing

                # --- spawn crop ---
                req = SpawnEntity.Request()
                req.name = f'crop_{crop_idx}'
                req.xml  = black_sdf
                req.robot_namespace = ''
                req.initial_pose.position.x = x_pos
                req.initial_pose.position.y = y_pos
                req.initial_pose.position.z = 0.002
                future = self.cli.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                crop_idx += 1

        # Weeds placed strictly BETWEEN crop rows
        for gap in range(num_rows - 1):
            y_between = start_y + gap * row_spacing + row_spacing / 2.0
            for col in range(num_cols):
                x_pos = start_x + col * col_spacing
                if random.random() < 0.30:   # 30 % chance per gap cell
                    wx = x_pos + random.uniform(-0.05, 0.05)
                    wy = y_between + random.uniform(-0.08, 0.08)

                    req_w = SpawnEntity.Request()
                    req_w.name = f'weed_{weed_idx}'
                    req_w.xml  = red_sdf
                    req_w.robot_namespace = ''
                    req_w.initial_pose.position.x = wx
                    req_w.initial_pose.position.y = wy
                    req_w.initial_pose.position.z = 0.002
                    future = self.cli.call_async(req_w)
                    rclpy.spin_until_future_complete(self, future)

                    weed_map[f'weed_{weed_idx}'] = {'x': wx, 'y': wy}
                    weed_idx += 1

        # Publish the weed map on the latched topic
        msg = String()
        msg.data = json.dumps(weed_map)
        self.weed_pub.publish(msg)
        self.get_logger().info(
            f'Spawned {crop_idx} crops and {weed_idx} weeds. Map published.')


def main():
    rclpy.init()
    node = FieldSpawner()
    # Keep spinning briefly so the latched message is sent
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
