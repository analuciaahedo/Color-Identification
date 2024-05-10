import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class CVExample(Node):
    def __init__(self):
        super().__init__('cv_node')
        
        self.img_video = np.ndarray((720, 1280, 3))
        self.rojo_lower = np.array([136, 87, 111], np.uint8) 
        self.rojo_upper = np.array([180, 255, 255], np.uint8)

        self.valid_img = False
        self.bridge = CvBridge()

        self.video_sub = self.create_subscription(Image, '/video_source/raw', self.camera_callback, 10)
        self.red_pub = self.create_publisher(Image, 'img_properties/red/msk', 10)

        dt = 0.1
        self.timer = self.create_timer(dt, self.timer_callback)
        self.get_logger().info('CV Node started')

    def camera_callback(self, msg):
        try:
            self.img_video = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.valid_img = True
        except:
            self.get_logger().info('Failed to get an image')

    def timer_callback(self):
        try:
            if self.valid_img:
                hsv_img = cv2.cvtColor(self.img_video, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_img, self.rojo_lower, self.rojo_upper)
                red_output = cv2.bitwise_and(self.img_video, self.img_video, mask=mask)

                kernel_red = np.ones((5, 5), np.uint8)
                dilatacion = cv2.dilate(red_output, kernel_red, iterations=1)
                erosion = cv2.erode(dilatacion, kernel_red, iterations=1)


                gray = cv2.cvtColor(self.img_video, cv2.COLOR_BGR2GRAY)
                channel_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                channel_gray[:, :, 2] = np.maximum(erosion[:, :, 2], channel_gray[:, :, 2]) 

                self.red_pub.publish(self.bridge.cv2_to_imgmsg(channel_gray, "bgr8"))
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
