import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class CVExample(Node):
    def __init__(self):
        super().__init__('cv_node')
        
        self.img = np.ndarray((720, 1280, 3))
        self.lower = np.array([100, 50, 50])
        self.upper = np.array([130, 255, 255])

        self.valid_img = False
        self.bridge = CvBridge()

        self.sub = self.create_subscription(Image, '/video_source/raw', self.camera_callback, 10)
        self.pub = self.create_publisher(Image, '/img_processing/color', 10)

        dt = 0.1
        self.timer = self.create_timer(dt, self.timer_callback)
        self.get_logger().info('CV Node started')

    def camera_callback(self, msg):
        try:
            self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.valid_img = True
        except:
            self.get_logger().info('Failed to get an image')

    def timer_callback(self):
        try:
            if self.valid_img:
               hsvFrame = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
               mask = cv2.inRange(hsvFrame, self.lower, self.upper)
               detected_output = cv2.bitwise_and(self.img, self.img, mask=mask)
               gray = cv2.cvtColor(detected_output, cv2.COLOR_BGR2GRAY)
               
               #blue_detection = cv2.bitwise_and(self.img, self.img, mask=mask)

               #blue_mask = cv2.cvtColor(detected_output, cv2.COLOR_BGR2GRAY)
              # self.img[blue_mask > 0] = [255, 0, 0]  # Azul
               gray_with_blue_highlight = cv2.addWeighted(gray, 1, detected_output, 0.7, 0)
                
                # Publicar la imagen procesada
               self.pub.publish(self.bridge.cv2_to_imgmsg(gray_with_blue_highlight, "mono8"))
        except:
            self.get_logger().info('Failed to process image')

def main(args=None):
    rclpy.init(args=args)
    cv_e = CVExample()
    rclpy.spin(cv_e)
    cv_e.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
