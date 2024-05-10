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
               # Convertir a escala de grises
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Convertir a HSV
                hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                # Aplicar la máscara azul
                lower_blue = np.array([100, 50, 50])  # Rango de color azul en HSV
                upper_blue = np.array([130, 255, 255])
                mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
                blue_detection = cv2.bitwise_and(img, img, mask=mask)

                # Convertir la detección de azul a azul en la imagen
                blue_mask = cv2.cvtColor(blue_detection, cv2.COLOR_BGR2GRAY)
                img[blue_mask > 0] = [255, 0, 0]  # Azul
                
                # Publicar la imagen procesada
                self.pub.publish(self.bridge.cv2_to_imgmsg(gray, "mono8"))
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
