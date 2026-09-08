#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class SerialBridgeNode(Node):
    def __init__(self):
        super().__init__('serial_bridge_node')
        
        self.sub_state = self.create_subscription(
            String,
            '/weeder/state',
            self.state_callback,
            10)
            
        self.sub_cmd_vel = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)
            
        self.get_logger().info("Serial Bridge Node Started - Mocking ESP32 Comm")
        
        # State tracking so we don't spam the console too much
        self.current_state = ""

    def state_callback(self, msg):
        if msg.data != self.current_state:
            self.current_state = msg.data
            packet = f"<LED_CMD:{self.current_state}>"
            self.send_to_serial(packet)

    def cmd_vel_callback(self, msg):
        v = msg.linear.x
        w = msg.angular.z
        
        # Simple differential drive kinematics mock
        base_width = 0.23
        wheel_radius = 0.05
        
        v_l = v - (w * base_width / 2.0)
        v_r = v + (w * base_width / 2.0)
        
        # Convert to pretend PWM or RPM
        rpm_l = int((v_l / (2 * 3.14159 * wheel_radius)) * 60)
        rpm_r = int((v_r / (2 * 3.14159 * wheel_radius)) * 60)
        
        packet = f"<MOTOR_CMD:{rpm_l},{rpm_r}>"
        self.send_to_serial(packet)

    def send_to_serial(self, packet):
        # In the future, this would be: self.serial_port.write(packet.encode('utf-8'))
        self.get_logger().info(f"ESP32 TX -> {packet}")

def main(args=None):
    rclpy.init(args=args)
    node = SerialBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
