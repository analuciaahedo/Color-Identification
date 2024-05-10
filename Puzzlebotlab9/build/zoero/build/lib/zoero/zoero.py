import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import Point

class CVExample(Node):
    def __init__(self):
        super().__init__('cv_node')
        
        self.img = np.ndarray((720, 1280, 3))
        # Inicializa los valores para el color rojo
        self.lower = np.array([136, 87, 111])
        self.upper = np.array([180, 255, 255])

        self.valid_img = False
        self.bridge = CvBridge()
        
        # Nos suscribimos al tópico del video 
        self.sub = self.create_subscription(Image, '/video_source/raw', self.camera_callback, 10)
        
        # Creamos el publicador para la imagen procesada
        self.pub_processing = self.create_publisher(Image, '/img_processing/color', 10)
        
        # Creamos el publicador para la máscara roja
        self.pub_red_mask = self.create_publisher(Image, 'red/mask', 10)

        # Creamos la función del temporizador 
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('CV Node started')

        # Publicadores para las propiedades del color rojo
        self.publisher_mask_red = self.create_publisher(Image, 'img_properties/red/msk', 10)
        self.publisher_density_red = self.create_publisher(Float32, 'img_properties/red/density', 10)
        self.publisher_xy_red = self.create_publisher(Point, 'img_properties/red/xy', 10)

    def camera_callback(self, msg):
        try:
            self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.valid_img = True
        except Exception as e:
            self.get_logger().info(f'Failed to get an image: {e}')

    def timer_callback(self):
        try:
            if not self.valid_img:
                return  # Retorna si la imagen no es válida

            hsvFrame = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            # Crea la máscara para el color rojo
            mask = cv2.inRange(hsvFrame, self.lower, self.upper)
            detected_output = cv2.bitwise_and(self.img, self.img, mask=mask)

            # Convertir la imagen a escala de grises y aplicar un umbral
            gray_img = cv2.cvtColor(detected_output, cv2.COLOR_BGR2GRAY)
            _, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

            # Dilatar y erosionar la imagen para resaltar las características
            kernel = np.ones((5, 5), np.uint8)
            img_dilated_red = cv2.dilate(thresh_img, kernel, iterations=1)
            img_eroded_red = cv2.erode(img_dilated_red, kernel, iterations=1)

            # Publicar la máscara roja
            self.pub_red_mask.publish(self.bridge.cv2_to_imgmsg(img_eroded_red))

            # Calcular la densidad de píxeles rojos y las coordenadas del centro de masa
            total_pixels_red = np.prod(mask.shape[:2])
            red_pixels = cv2.countNonZero(mask)
            if total_pixels_red > 0:
                red_density = red_pixels / total_pixels_red
                M_red = cv2.moments(mask)
                center_x_red = int(M_red["m10"] / M_red["m00"])
                center_y_red = int(M_red["m01"] / M_red["m00"])
            else:
                red_density = 0.0
                center_x_red, center_y_red = 0, 0

            # Publicar mensajes de densidad y coordenadas
            density_red_msg = Float32()
            density_red_msg.data = red_density
            xy_red_msg = Point()
            xy_red_msg.x = center_x_red
            xy_red_msg.y = center_y_red

            self.publisher_density_red.publish(density_red_msg)
            self.publisher_xy_red.publish(xy_red_msg)
                
        except Exception as e:
            self.get_logger().info(f'Failed to process image: {e}')

def main(args=None):
    rclpy.init(args=args)
    cv_e = CVExample()
    try:
        rclpy.spin(cv_e)
    except KeyboardInterrupt:
        pass
    cv_e.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

