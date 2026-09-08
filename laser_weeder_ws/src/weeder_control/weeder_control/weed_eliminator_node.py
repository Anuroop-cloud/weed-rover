#!/usr/bin/env python3
"""
Weed Eliminator Node

Reliable approach:
- Robot position: /odom  (always published by the diff_drive plugin)
- Weed positions: /weeder/weed_map  (latched topic from spawn_field.py)
- Deletion: /delete_entity  Gazebo service
"""
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSProfile,
                       QoSReliabilityPolicy, QoSHistoryPolicy)

from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from gazebo_msgs.srv import DeleteEntity


class WeederControlNode(Node):

    def __init__(self):
        super().__init__('weeder_control_node')

        # ── QoS profiles ────────────────────────────────────────────
        reliable_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        # ── Subscriptions ────────────────────────────────────────────
        # Robot position via odometry (ALWAYS published by diff_drive)
        self.create_subscription(Odometry, '/odom',
                                 self.odom_callback, reliable_qos)

        # Weed map published (once, latched) by spawn_field.py
        self.create_subscription(String, '/weeder/weed_map',
                                 self.weed_map_callback, latched_qos)

        # Red-weed detections from perception node
        self.create_subscription(Point, '/weeder/detected_weed',
                                 self.detection_callback, reliable_qos)

        # ── Publishers ───────────────────────────────────────────────
        self.pub_markers = self.create_publisher(
            MarkerArray, '/weeder/led_matrix_markers', 10)
        self.pub_state = self.create_publisher(String, '/weeder/state', 10)

        # ── Service client ───────────────────────────────────────────
        self.cli_delete = self.create_client(DeleteEntity, '/delete_entity')

        # ── State ────────────────────────────────────────────────────
        self.state = 'SEARCHING'
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.weed_poses: dict = {}          # name -> (x, y)
        self.last_detect_time = 0.0
        self.last_detect_pos  = (0.0, 0.0)  # normalised (nx, ny) in camera
        self.cooldown_until   = 0.0
        self.flash_until      = 0.0

        self.create_timer(0.1, self.loop)
        self.get_logger().info('WeederControlNode started – waiting for weed map…')

    # ── Callbacks ────────────────────────────────────────────────────

    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def weed_map_callback(self, msg: String):
        data = json.loads(msg.data)
        self.weed_poses = {k: (v['x'], v['y']) for k, v in data.items()}
        self.get_logger().info(
            f'Weed map received: {len(self.weed_poses)} weeds tracked.')

    def detection_callback(self, msg: Point):
        now = time.time()
        self.last_detect_time = now
        self.last_detect_pos  = (msg.x, msg.y)

        if self.state == 'SEARCHING':
            self.state = 'TARGET_DETECTED'
            self.get_logger().info('Weed detected → TARGET_DETECTED')

    # ── Main loop ─────────────────────────────────────────────────────

    def loop(self):
        now = time.time()

        if self.state == 'TARGET_DETECTED':
            if now - self.last_detect_time > 0.8:
                self.state = 'SEARCHING'
                self.get_logger().info('Target lost → SEARCHING')
            elif now > self.cooldown_until:
                nx, ny = self.last_detect_pos
                # Fire when the weed blob is within the centre 60 % of the frame
                if abs(nx) < 0.6:
                    self.state = 'ELIMINATION_TRIGGERED'
                    self.get_logger().info(
                        f'Laser FIRE at cam ({nx:.2f}, {ny:.2f})')

        elif self.state == 'ELIMINATION_TRIGGERED':
            self._eliminate()
            self.flash_until   = now + 0.6
            self.cooldown_until = now + 2.0
            self.state = 'MOVING_TO_NEXT_TARGET'

        elif self.state == 'MOVING_TO_NEXT_TARGET':
            if now - self.last_detect_time > 1.5:
                self.state = 'SEARCHING'

        # Publish state string
        sm = String(); sm.data = self.state
        self.pub_state.publish(sm)

        self._publish_markers(now)

    # ── Elimination ───────────────────────────────────────────────────

    def _eliminate(self):
        if not self.weed_poses:
            self.get_logger().warn('Weed map is empty – nothing to delete.')
            return

        # Find nearest weed using own tracked positions
        nearest = None
        min_d   = float('inf')
        for name, (wx, wy) in self.weed_poses.items():
            d = math.hypot(wx - self.robot_x, wy - self.robot_y)
            if d < min_d:
                min_d   = d
                nearest = name

        self.get_logger().info(
            f'Nearest weed: {nearest} at {min_d:.3f} m  '
            f'(robot @ {self.robot_x:.2f}, {self.robot_y:.2f})')

        if nearest is None or min_d > 1.0:
            self.get_logger().warn(
                f'No weed within 1.0 m (closest {min_d:.2f} m)')
            return

        self.get_logger().info(f'Deleting {nearest}')
        req = DeleteEntity.Request()
        req.name = nearest

        if self.cli_delete.wait_for_service(timeout_sec=2.0):
            future = self.cli_delete.call_async(req)
            future.add_done_callback(
                lambda f: self.get_logger().info(
                    f'Delete result: {f.result().success} – {f.result().status_message}'))
            # Remove from local map immediately
            self.weed_poses.pop(nearest, None)
        else:
            self.get_logger().error('/delete_entity service not available!')

    # ── Markers ───────────────────────────────────────────────────────

    def _led_color(self):
        return {
            'SEARCHING':            (0.2, 0.2, 0.2),
            'TARGET_DETECTED':      (1.0, 1.0, 0.0),
            'ELIMINATION_TRIGGERED':(1.0, 0.0, 0.0),
            'MOVING_TO_NEXT_TARGET':(0.0, 1.0, 0.0),
        }.get(self.state, (0.2, 0.2, 0.2))

    def _publish_markers(self, now: float):
        ma    = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        r, g, b = self._led_color()

        for idx, frame in enumerate(('led_matrix_left', 'led_matrix_right')):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp    = stamp
            m.ns  = 'led'; m.id = idx
            m.type   = Marker.CUBE
            m.action = Marker.ADD
            m.scale.x = 0.05; m.scale.y = 0.04; m.scale.z = 0.005
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.9
            ma.markers.append(m)

        flash = Marker()
        flash.header.frame_id = 'base_footprint'
        flash.header.stamp    = stamp
        flash.ns = 'laser_flash'; flash.id = 2

        if now < self.flash_until:
            flash.type   = Marker.CYLINDER
            flash.action = Marker.ADD
            flash.pose.position.x = 0.22
            flash.pose.position.z = 0.003
            flash.scale.x = 0.25; flash.scale.y = 0.25; flash.scale.z = 0.01
            flash.color.r = 1.0; flash.color.a = 0.95
        else:
            flash.action = Marker.DELETE

        ma.markers.append(flash)
        self.pub_markers.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(WeederControlNode())


if __name__ == '__main__':
    main()
