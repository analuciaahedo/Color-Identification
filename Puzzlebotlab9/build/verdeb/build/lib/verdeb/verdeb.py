import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class CVExample(Node):
    def __init__(self):
        super().__init__('cv_node')
        
        self.img = np.ndarray((720, 1280, 3))
        self.lower = np.array([25, 52, 72], np.uint8) 
        self.upper = np.array([102, 255, 255], np.uint8)

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

                # Dilatación y erosión para acentuar características
                kernel = np.ones((5, 5), np.uint8)
                dilated_image = cv2.dilate(detected_output, kernel, iterations=1)
                eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

                # Crear la imagen en escala de grises con los detalles en rojo visibles
                gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
                three_channel_gray = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # Convierte a BGR
                three_channel_gray[:, :, 1] = np.maximum(eroded_image[:, :, 1], three_channel_gray[:, :, 1])  # Combina el rojo con gris

                self.pub.publish(self.bridge.cv2_to_imgmsg(three_channel_gray, "bgr8"))
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

