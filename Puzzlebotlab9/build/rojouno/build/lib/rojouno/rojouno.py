import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
class CVExample(Node):
    def __init__(self):
        super().__init__('cv_node')
        
        self.img = np.ndarray((720, 1280, 3))  # Inicializa una matriz para almacenar la imagen
        self.lower = np.array([136, 87, 111])   # Valor inferior del rango de color HSV
        self.upper = np.array([180, 255, 255])  # Valor superior del rango de color HSV

        self.valid_img = False  # Bandera para indicar si se ha recibido una imagen válida
        self.bridge = CvBridge()  # Objeto para realizar la conversión entre tipos de datos OpenCV y ROS

        # Suscriptor para recibir imágenes de la cámara desde el tópico '/image_raw'
        self.sub = self.create_subscription(Image, '/image_raw', self.camera_callback, 10)
        # Publicador para publicar imágenes procesadas en el tópico '/img_processing/color'
        self.pub = self.create_publisher(Image, '/img_processing/color', 10)

        dt = 0.1
        # Temporizador para llamar a la función de callback a intervalos regulares
        self.timer = self.create_timer(dt, self.timer_callback)
        self.get_logger().info('CV Node started')

    def camera_callback(self, msg):
        try:
            # Convierte el mensaje de imagen a una matriz de OpenCV
            self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.valid_img = True
        except:
            self.get_logger().info('Failed to get an image')

    def timer_callback(self):
        try:
            if self.valid_img:
                # Convierte la imagen de BGR a HSV
                hsvFrame = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
                # Crea una máscara para los píxeles que están dentro del rango de color definido
                mask = cv2.inRange(hsvFrame, self.lower, self.upper)
                #maybe con color
                y3  = cv2.bitwise_and(self.img,self.img,mask=mask)
                # Aplica la máscara a la imagen original
                detected_output = cv2.bitwise_and(self.img, self.img, mask=mask)
                # Convierte la imagen a escala de grises
                gray = cv2.cvtColor(detected_output, cv2.COLOR_BGR2GRAY)
                # Aplica un desenfoque para suavizar la imagen
                blur = cv2.medianBlur(gray, 5)
                # Aplica el algoritmo de detección de bordes Canny
                canny = cv2.Canny(blur, 75, 250)
                # Publica la imagen procesada en el tópico '/img_processing/color'
                self.pub.publish(self.bridge.cv2_to_imgmsg(y3))
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

