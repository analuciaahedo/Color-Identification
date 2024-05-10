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
        #inicializan los colores para rojo
        self.lower = np.array([136, 87, 111])
        self.upper = np.array([180, 255, 255])

        self.valid_img = False
        self.bridge = CvBridge()
        #nos suscribimos al topico del video 
        self.sub = self.create_subscription(Image, '/video_source/raw', self.camera_callback, 10)
        #creamos el publiclador del primer topico donde se procesa la imagen
        self.pub = self.create_publisher(Image, '/img_processing/color', 10)
        #creamos el publicador de la segunda imangen 
        self.pub = self.create_publisher(Image,'red/mask',10)
        #creamos la funcion del timer 
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('CV Node started')
        #Rojo
        self.publisher_mask_red = self.create_publisher(Image, 'img_properties/red/msk', 10)
        self.publisher_density_red = self.create_publisher(Float32, 'img_properties/red/density', 10)
        self.publisher_xy_red = self.create_publisher(Point, 'img_properties/red/xy', 10)

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
                #crea la mascara
                mask_red= cv2.inRange(hsvFrame, self.lower, self.upper)
                detected_output = cv2.bitwise_and(self.img, self.img, mask_red=mask_red)
        # Cambia la imagen a escala de grises y aplica un threshold
                gray_img = cv2.cvtColor(detected_output, cv2.COLOR_BGR2GRAY)
                _, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
             # Dilata y erosiona la imagen usando un kernel de 5x5 para acentuar las características de la imagen.
                kernel = np.ones((5, 5), np.uint8)
                img_dilatada_red = cv2.dilate(thresh_img, kernel, iterations=1)
                img_erosionada_red = cv2.erode(img_dilatada_red, kernel, iterations=1)

                # Publicar la imagen
                #processed_red_msg = (self.bridge.cv2_to_imgmsg(img_erosionada_red))
                self.publisher_mask_red.publish(self.bridge.cv2_to_imgmsg(img_erosionada_red))

                #self.pub.publish(self.bridge.cv2_to_imgmsg(canny))

                # Detecta qué porcentaje de la imagen es de color rojo.
                total_pixels_red = np.prod(mask_red.shape[:2])
                red_pixels = cv2.countNonZero(mask_red)
                red_density = red_pixels / total_pixels_red

                 # Calcula el centro de masa en 𝑥 y 𝑦 de la detección en rojo.
                M_red = cv2.moments(mask_red)
                if M_red["m00"] != 0:
                    center_x_red = int(M_red["m10"] / M_red["m00"])
                    center_y_red = int(M_red["m01"] / M_red["m00"])
                else:
                    center_x_red, center_y_red = 0, 0

                #mask_red_msg = self.bridge.cv2_to_imgmsg(mask_red, "mono8")
                density_red_msg = Float32()
                density_red_msg.data = red_density
                xy_red_msg = Point()
                xy_red_msg.x = center_x_red
                xy_red_msg.y = center_y_red

                self.publisher_mask_red.publish(self.bridge.cv2_to_imgmsg(mask_red))
                self.publisher_density_red.publish(density_red_msg)
                self.publisher_xy_red.publish(xy_red_msg)


                
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
