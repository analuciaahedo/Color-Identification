import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class CVExample(Node):
    def __init__(self):
        super().__init__('cv_node')
        
        self.video_green = np.ndarray((720, 1280, 3))
        self.verde_lower = np.array([60, 50, 50], np.uint8) 
        self.verde_upper = np.array([120, 255, 255], np.uint8)

        self.valid_img = False
        self.bridge = CvBridge()

        self.svideo_sub = self.create_subscription(Image, '/video_source/raw', self.camera_callback, 10)
        self.verde_pub = self.create_publisher(Image, 'img_properties/green/msk', 10)

        dt = 0.1
        self.timer = self.create_timer(dt, self.timer_callback)
        self.get_logger().info('CV Node started')

    def camera_callback(self, msg):
        try:
            self.video_green = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.valid_img = True
        except:
            self.get_logger().info('Failed to get an image')

    def timer_callback(self):
        try:
            if self.valid_img:
                hsvFrame = cv2.cvtColor(self.video_green, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsvFrame, self.verde_lower, self.verde_upper)
                verde_output = cv2.bitwise_and(self.video_green, self.video_green, mask=mask)

                # Dilatación y erosión para acentuar características
                kernel = np.ones((5, 5), np.uint8)
                dilated_verde = cv2.dilate(verde_output, kernel, iterations=1)
                eroded_verde = cv2.erode(dilated_verde, kernel, iterations=1)

                # Crear la imagen en escala de grises con los detalles en rojo visibles
                gray_verde = cv2.cvtColor(self.video_green, cv2.COLOR_BGR2GRAY)
                verde_channel_gray = cv2.cvtColor(gray_verde, cv2.COLOR_GRAY2BGR)  # Convierte a BGR
                verde_channel_gray[:, :, 1] = np.maximum(eroded_verde[:, :, 1], verde_channel_gray[:, :, 1])  # Combina el rojo con verde

                self.verde_pub.publish(self.bridge.cv2_to_imgmsg(verde_channel_gray, "bgr8"))
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

