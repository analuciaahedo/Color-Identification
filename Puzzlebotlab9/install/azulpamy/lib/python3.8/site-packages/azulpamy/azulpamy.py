import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

class CVExample(Node):
    def __init__(self):
        super().__init__('cv_node')
        
        self.img = np.ndarray((720, 1280, 3))
        # Definimos el rango para el color azul en HSV
        self.lower_blue = np.array([100, 150, 50], np.uint8)  # Azul claro
        self.upper_blue = np.array([140, 255, 255], np.uint8)  # Azul oscuro

        self.valid_img = False
        self.bridge = CvBridge()

        self.sub = self.create_subscription(Image, '/video_source/raw', self.camera_callback, 10)
        self.pub = self.create_publisher(Float32, '/img_properties/blue/density', 10)

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
        if self.valid_img:
            hsvFrame = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            # Crear máscaras para el azul
            blue_mask = cv2.inRange(hsvFrame, self.lower_blue, self.upper_blue)

            cv2.imshow("Blue Detection Mask", blue_mask)
            cv2.waitKey(1)

            # Contar el número de píxeles azules
            blue_pixel_count = np.sum(blue_mask > 0)  # Los píxeles detectados como azules
            total_pixels = self.img.shape[0] * self.img.shape[1]
            blue_density = blue_pixel_count / total_pixels

            # Definir un umbral para considerar que la imagen contiene color azul
            blue_threshold = 0.01  # 1% de los píxeles

            if blue_density < blue_threshold:
                density_percentage = blue_density / blue_threshold
            else:
                density_percentage = 1.0

def main(args=None):
    rclpy.init(args=args)
    cv_example = CVExample()
    rclpy.spin(cv_example)
    cv_example.destroy_node()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

