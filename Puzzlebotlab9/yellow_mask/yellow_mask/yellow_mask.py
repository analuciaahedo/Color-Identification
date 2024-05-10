import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class CVExample(Node):
    def __init__(self):
        super().__init__('cv_node')
        
        self.ama_img = np.ndarray((720, 1280, 3))
        self.ama_lower = np.array([20,100,100], np.uint8) 
        self.ama_upper = np.array([30,255,255], np.uint8)

        self.valid_img = False
        self.bridge = CvBridge()

        self.video_subs = self.create_subscription(Image, '/video_source/raw', self.camera_callback, 10)
        self.ama_pub = self.create_publisher(Image, 'img_properties/yellow/msk', 10)

        dt = 0.1
        self.timer = self.create_timer(dt, self.timer_callback)
        self.get_logger().info('CV Node started')

    def camera_callback(self, msg):
        try:
            self.ama_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.valid_img = True
        except:
            self.get_logger().info('Failed to get an image')

    def timer_callback(self):
        try:
            if self.valid_img:
                hsvFrame = cv2.cvtColor(self.ama_img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsvFrame, self.ama_lower, self.ama_upper)
                ama_output = cv2.bitwise_and(self.ama_img, self.ama_img, mask=mask)

                # Dilatación y erosión para acentuar características
                kernel = np.ones((5, 5), np.uint8)
                dilated_ama = cv2.dilate(ama_output, kernel, iterations=1)
                eroded_ama = cv2.erode(dilated_ama, kernel, iterations=1)

                # Crear la imagen en escala de grises con los detalles en rojo visibles
                gray_ama = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
                ama_channel_gray = cv2.cvtColor(gray_ama, cv2.COLOR_GRAY2BGR)  # Convierte a BGR
                #three_channel_gray[:, :, 1] = np.maximum(eroded_image[:, :, 1], three_channel_gray[:, :, 1])  # Combina el rojo con verde
                ama_channel_gray[:, :, 0] = np.maximum(eroded_ama[:, :, 0], ama_channel_gray[:, :, 0])  # Combina el amarillo con gris
                ama_channel_gray[:, :, 1] = np.maximum(eroded_ama[:, :, 1], ama_channel_gray[:, :, 1])  # Combina el amarillo con gris
                ama_channel_gray[:, :, 2] = np.maximum(eroded_ama[:, :, 2], ama_channel_gray[:, :, 2])

                self.ama_pub.publish(self.bridge.cv2_to_imgmsg(ama_channel_gray, "bgr8"))
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

