import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge #Conversion de tipo de dato CV2 a Ros2
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
import cv2
import numpy as np

class ImageIdentification(Node):
    def _init_(self):
        super()._init_('image:processor')
        self.publisher_mask = self.create_publisher(Image, 'img_properties/red/msk', 10)
        self.publisher_density = self.create_publisher(Float32, 'img_properties/red/density', 10)
        self.publisher_xy = self.create_publisher(Point, '/img_properties/red/xy', 10)
        self.publisher_processed_img = self.create_publisher(Image, '/img_properties/red/processed_img', 10)
        self.subscription = self.create_subscription(Image, 'video_source/raw', self.process_image, 10)
        self.timer = self.create_timer(0.1, self.process_image)
        self.cv_bridge = CvBridge()

    def process_image(self, msg):
        cv_img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV)
        
        # Detecta el color rojo de una imagen usando HSV.
        light_red = np.array([0, 50, 50])
        dark_red = np.array([15, 255, 255])
        # También necesitas considerar el rango de 165 a 180 para el rojo
        light_red1 = np.array([165, 50, 50])
        dark_red1 = np.array([180, 255, 255])

        
        # Crea una máscara.
        mask = cv2.inRange(hsv_img, dark_red, light_red,light_red1,dark_red1)
        
        # Cambia la imagen a escala de grises y aplica un threshold
        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        _, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

        # Dilata y erosiona la imagen usando un kernel de 5x5 para acentuar las características de la imagen.
        kernel = np.ones((5, 5), np.uint8)
        img_dilatada = cv2.dilate(thresh_img, kernel, iterations=1)
        img_erosionada = cv2.erode(img_dilatada, kernel, iterations=1)

        # Publicar la imagen
        processed_msg = self.cv_bridge.cv2_to_imgmsg(img_erosionada, "mono8")
        self.publisher.publish(processed_msg)

        # Detecta qué porcentaje de la imagen es de color rojo.
        total_pixels = np.prod(mask.shape[:2])
        red_pixels = cv2.countNonZero(mask)
        red_density = red_pixels / total_pixels

        # Calcula el centro de masa en 𝑥 y 𝑦 de la detección en rojo.
        M = cv2.moments(mask)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x, center_y = 0, 0

        mask_msg = self.cv_bridge.cv2_to_imgmsg(mask, "mono8")
        density_msg = Float32()
        density_msg.data = red_density
        xy_msg = Point()
        xy_msg.x = center_x
        xy_msg.y = center_y

        self.publisher_mask.publish(mask_msg)
        self.publisher_density.publish(density_msg)
        self.publisher_xy.publish(xy_msg)


def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageIdentification()
    rclpy.spin(image_processor)
    image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
