#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
import cv2
from cv_bridge import CvBridge
import numpy as np
from rclpy.qos import qos_profile_sensor_data

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data)
        self.publisher = self.create_publisher(Point, '/weeder/detected_weed', 10)
        
        # Publish an annotated image for visualization in RViz
        self.image_pub = self.create_publisher(Image, '/weeder/annotated_image', 10)
        self.bridge = CvBridge()
        self.get_logger().info("Perception Node Started")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return
            
        # Convert to HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Define range for red color in HSV
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width, _ = cv_image.shape
        center_x = width // 2
        center_y = height // 2
        
        for contour in contours:
            if cv2.contourArea(contour) > 20:  # Filter small noise
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Draw for visualization
                    cv2.circle(cv_image, (cx, cy), 10, (0, 255, 0), 2)
                    
                    # Simple heuristic: calculate coordinates relative to camera frame
                    # assuming flat ground and fixed camera.
                    # This is a mock calculation suitable for simulation.
                    # x is forward/backward, y is left/right in camera optical frame
                    
                    # Publish the target point in pixel space for now, 
                    # or approximated metric space.
                    # Let's normalize it to [-1.0, 1.0] for easier handling by control node
                    norm_x = (cx - center_x) / center_x
                    norm_y = (cy - center_y) / center_y
                    
                    p = Point()
                    p.x = float(norm_x)
                    p.y = float(norm_y)
                    p.z = 0.0 # Unused
                    self.publisher.publish(p)
                    
                    # Log detection (useful for debugging)
                    # self.get_logger().info(f"Red weed detected at normalized ({norm_x:.2f}, {norm_y:.2f})")
                    
        # Publish annotated image
        annotated_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.image_pub.publish(annotated_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
