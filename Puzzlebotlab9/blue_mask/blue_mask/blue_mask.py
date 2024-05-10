import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class CVExample(Node):
    def __init__(self):
        super().__init__('cv_node')
        
        self.video = np.ndarray((720, 1280, 3))
        self.azul_lower = np.array([94, 80, 2], np.uint8) 
        self.azul_upper = np.array([120, 255, 255], np.uint8)

        self.valid_img = False
        self.bridge = CvBridge()

        self.video_sub = self.create_subscription(Image, '/video_source/raw', self.camera_callback, 10)
        self.azul_pub = self.create_publisher(Image, 'img_properties/blue/msk', 10)

        dt = 0.1
        self.timer = self.create_timer(dt, self.timer_callback)
        self.get_logger().info('CV Node started')

    def camera_callback(self, msg):
        try:
            self.video = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.valid_img = True
        except:
            self.get_logger().info('Failed to get an image')

    def timer_callback(self):
        try:
            if self.valid_img:
                hsv_img = cv2.cvtColor(self.video, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_img, self.azul_lower, self.azul_upper)
                azul_output = cv2.bitwise_and(self.video, self.video, mask=mask)


                kernel = np.ones((5, 5), np.uint8)
                dilated_azul = cv2.dilate(azul_output, kernel, iterations=1)
                eroded_azul = cv2.erode(dilated_azul, kernel, iterations=1)


                gray_azul = cv2.cvtColor(self.video, cv2.COLOR_BGR2GRAY)
                azul_channel_gray = cv2.cvtColor(gray_azul, cv2.COLOR_GRAY2BGR)  
                azul_channel_gray[:, :, 0] = np.maximum(eroded_azul[:, :, 0], azul_channel_gray[:, :, 0]) 

                self.azul_pub.publish(self.bridge.cv2_to_imgmsg(azul_channel_gray, "bgr8"))
        except:
            self.get_logger().info('Failed to process image')

def main(args=None):
    rclpy.init(args=args)
    cv_example = CVExample()
    rclpy.spin(cv_example)
    cv_example.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

