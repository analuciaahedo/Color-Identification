import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import Point

class CVExample(Node):
    def __init__(self):
        super().__init__('cv_node')
        self.img = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.valid_img = False
        self.bridge = CvBridge()

        # Suscribirse al tópico de la fuente de video
        self.sub = self.create_subscription(Image, '/video_source/raw', self.camera_callback, 10)

        # Creación de tópicos para color rojo
        self.pub_red = self.create_publisher(Image, 'img_properties/red/msk', 10)
        self.pubper_red = self.create_publisher(Float32, '/img_properties/red/percentage', 10)
        self.publisher_xy_red = self.create_publisher(Point, '/img_properties/red/xy', 10)

        # Creación de tópicos para color azul
        self.pub_blue = self.create_publisher(Image, 'img_properties/blue/msk', 10)
        self.pubper_blue = self.create_publisher(Float32, '/img_properties/blue/percentage', 10)
        self.publisher_xy_blue = self.create_publisher(Point, '/img_properties/blue/xy', 10)

        #verde
        self.pub_green = self.create_publisher(Image, 'img_properties/green/msk', 10)
        self.pubper_green = self.create_publisher(Float32, '/img_properties/green/percentage', 10)
        self.publisher_xy_green = self.create_publisher(Point, '/img_properties/green/xy', 10)

        #amarillo 
        self.pub_yellow = self.create_publisher(Image, 'img_properties/yellow/msk', 10)
        self.pubper_yellow= self.create_publisher(Float32, '/img_properties/yellow/percentage', 10)
        self.publisher_xy_yellow= self.create_publisher(Point, '/img_properties/yellow/xy', 10)


        #amarillo 
        self.pub_white= self.create_publisher(Image, 'img_properties/white/msk', 10)
        self.pubper_white= self.create_publisher(Float32, '/img_properties/white/percentage', 10)
        self.publisher_xy_white= self.create_publisher(Point, '/img_properties/whitexy', 10)



        # Temporizador con un intervalo de 0.1 segundos
        dt = 0.1
        self.timer = self.create_timer(dt, self.process_image)
        self.get_logger().info('CV Node started')

    def camera_callback(self, msg):
        try:
            self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.valid_img = True
        except Exception as e:
            self.get_logger().info(f'Failed to get an image: {str(e)}')

    def process_image(self):
        if self.valid_img:
            try:
                hsv_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

                # Procesar rojo
                red_mask = self.create_red_mask(hsv_image)
                self.process_red_mask(red_mask)

                # Procesar azul
                blue_mask = self.create_blue_mask(hsv_image)
                self.process_blue_mask(blue_mask)

                #procesamos verde
            
                green_mask = self.create_green_mask(hsv_image)
                self.process_green_mask(green_mask)

                #procesamos amarillo
            
                yellow_mask = self.create_yellow_mask(hsv_image)
                self.process_yellow_mask(yellow_mask)

                white_mask = self.create_white_mask(hsv_image)
                self.process_white_mask(white_mask)


            except Exception as e:
                self.get_logger().info(f'Failed to process image: {str(e)}')

    def create_red_mask(self, hsv_image):
        lower_red1 = np.array([0, 100, 100], np.uint8)
        upper_red1 = np.array([10, 255, 255], np.uint8)
        lower_red2 = np.array([160, 100, 100], np.uint8)
        upper_red2 = np.array([180, 255, 255], np.uint8)
        red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        return cv2.bitwise_or(red_mask1, red_mask2)

    def create_blue_mask(self, hsv_image):
        lower_blue = np.array([100, 150, 0], np.uint8)
        upper_blue = np.array([140, 255, 255], np.uint8)
        return cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    def create_green_mask(self, hsv_image):
        lower_green = np.array([60,50,50], np.uint8)
        upper_green = np.array([120,250,250], np.uint8)
        return cv2.inRange(hsv_image, lower_green, upper_green)
    
    def create_yellow_mask(self, hsv_image):
        lower_yellow = np.array([20,50,50], np.uint8)
        upper_yellow = np.array([60,250,250], np.uint8)
        return cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    
    def create_white_mask(self, hsv_image):
        lower_white = np.array([0,0,153], np.uint8)
        upper_white = np.array([179,60,255], np.uint8)
        return cv2.inRange(hsv_image, lower_white, upper_white)
    

    




    def process_red_mask(self, red_mask):
        # Operaciones morfológicas para mejorar la máscara
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)
        red_mask = cv2.erode(red_mask, kernel, iterations=1)

        # Crear imagen en escala de grises resaltando en rojo
        gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        highlighted = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        highlighted[:, :, 2] = np.maximum(red_mask, highlighted[:, :, 2])

        # Publicar la imagen con áreas resaltadas
        self.pub_red.publish(self.bridge.cv2_to_imgmsg(highlighted, "bgr8"))

        self.calculate_and_publish_red_percentage(red_mask)
        self.calculate_and_publish_red_mass_center(red_mask)

        cv2.imshow("Rojo", red_mask)
        cv2.imshow("Image con rojo", highlighted)
        cv2.waitKey(1)

    def process_blue_mask(self, blue_mask):
        # Operaciones morfológicas para mejorar la máscara
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.dilate(blue_mask, kernel, iterations=1)
        blue_mask = cv2.erode(blue_mask, kernel, iterations=1)

        # Crear imagen en escala de grises resaltando en azul
        gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        highlighted = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        highlighted[:, :, 0] = np.maximum(blue_mask, highlighted[:, :, 0])

        # Publicar la imagen con áreas resaltadas
        self.pub_green.publish(self.bridge.cv2_to_imgmsg(highlighted, "bgr8"))

        self.calculate_and_publish_blue_percentage(blue_mask)
        self.calculate_and_publish_blue_mass_center(blue_mask)

        cv2.imshow("Azul", blue_mask)
        cv2.imshow("Imagen con azul", highlighted)
        cv2.waitKey(1)

    def process_green_mask(self, green_mask):
        # Operaciones morfológicas para mejorar la máscara
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.dilate(green_mask, kernel, iterations=1)
        green_mask = cv2.erode(green_mask, kernel, iterations=1)

        # Crear imagen en escala de grises resaltando en azul
        gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        highlighted = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        highlighted[:, :, 1] = np.maximum(green_mask, highlighted[:, :, 1])

        # Publicar la imagen con áreas resaltadas
        self.pub_green.publish(self.bridge.cv2_to_imgmsg(highlighted, "bgr8"))

        self.calculate_and_publish_green_percentage(green_mask)
        self.calculate_and_publish_green_mass_center(green_mask)

        cv2.imshow("Verde", green_mask)
        cv2.imshow("Imagen con verde", highlighted)
        cv2.waitKey(1)

    def process_yellow_mask(self, yellow_mask):
        # Operaciones morfológicas para mejorar la máscara
        kernel = np.ones((5, 5), np.uint8)
        yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=1)
        yellow_mask = cv2.erode(yellow_mask, kernel, iterations=1)

        # Crear imagen en escala de grises resaltando en azul
        gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        highlighted = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        highlighted[:, :, 0] = np.maximum(yellow_mask, highlighted[:, :, 0])
        highlighted[:, :, 1] = np.maximum(yellow_mask, highlighted[:, :, 1])
        highlighted[:, :, 2] = np.maximum(yellow_mask, highlighted[:, :, 2])

        # Publicar la imagen con áreas resaltadas
        self.pub_yellow.publish(self.bridge.cv2_to_imgmsg(highlighted, "bgr8"))

        self.calculate_and_publish_yellow_percentage(yellow_mask)
        self.calculate_and_publish_yellow_mass_center(yellow_mask)

        cv2.imshow("Amarillo", yellow_mask)
        cv2.imshow("Imagen con amarillo", highlighted)
        cv2.waitKey(1)


    def process_white_mask(self, white_mask):
        # Operaciones morfológicas para mejorar la máscara
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.dilate(white_mask, kernel, iterations=1)
        white_mask = cv2.erode(white_mask, kernel, iterations=1)

        # Crear imagen en escala de grises resaltando en azul
        gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        highlighted = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        highlighted[:, :, 0] = np.maximum(white_mask, highlighted[:, :, 0])
        highlighted[:, :, 1] = np.maximum(white_mask, highlighted[:, :, 1])
        highlighted[:, :, 2] = np.maximum(white_mask, highlighted[:, :, 2])

        # Publicar la imagen con áreas resaltadas
        self.pub_white.publish(self.bridge.cv2_to_imgmsg(highlighted, "bgr8"))

        self.calculate_and_publish_white_percentage(white_mask)
        self.calculate_and_publish_white_mass_center(white_mask)

        cv2.imshow("Blanco", white_mask)
        cv2.imshow("Image con blanco ", highlighted)
        cv2.waitKey(1)







    def calculate_and_publish_red_percentage(self, red_mask):
        red_pixel_count = np.sum(red_mask > 0)
        total_pixels = self.img.shape[0] * self.img.shape[1]
        red_percentage = red_pixel_count / total_pixels
        self.pubper_red.publish(Float32(data=red_percentage))


    def calculate_and_publish_blue_percentage(self, blue_mask):
        blue_pixel_count = np.sum(blue_mask > 0)
        total_pixels = self.img.shape[0] * self.img.shape[1]
        blue_percentage = blue_pixel_count / total_pixels
        self.pubper_blue.publish(Float32(data=blue_percentage))

    def calculate_and_publish_green_percentage(self, green_mask):
        green_pixel_count = np.sum(green_mask > 0)
        total_pixels = self.img.shape[0] * self.img.shape[1]
        green_percentage = green_pixel_count / total_pixels
        self.pubper_green.publish(Float32(data=green_percentage))

    def calculate_and_publish_yellow_percentage(self, yellow_mask):
        yellow_pixel_count = np.sum(yellow_mask > 0)
        total_pixels = self.img.shape[0] * self.img.shape[1]
        yellow_percentage = yellow_pixel_count / total_pixels
        self.pubper_yellow.publish(Float32(data=yellow_percentage))


    def calculate_and_publish_white_percentage(self, white_mask):
        white_pixel_count = np.sum(white_mask > 0)
        total_pixels = self.img.shape[0] * self.img.shape[1]
        white_percentage = white_pixel_count / total_pixels
        self.pubper_white.publish(Float32(data=white_percentage))





    def calculate_and_publish_red_mass_center(self, red_mask):
         M = cv2.moments(red_mask)
         if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                # Dibujar un círculo en el centro de masa en la imagen
                cv2.circle(self.img, (center_x, center_y), 10, (0, 255, 0), -1)
         else:
                center_x, center_y = 0, 0  # Defaults in case no red is detected

         cv2.imshow("Mascara rojo", red_mask)
         cv2.imshow("<centro de masa", self.img)
         cv2.waitKey(1)

    def calculate_and_publish_blue_mass_center(self, blue_mask):
            M = cv2.moments(blue_mask)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                # Dibujar un círculo en el centro de masa en la imagen
                cv2.circle(self.img, (center_x, center_y), 10, (0, 255, 0), -1)
            else:
                center_x, center_y = 0, 0  # Defaults in case no red is detected

            cv2.imshow("Mascara azul", blue_mask)
            cv2.imshow("Centro de masa", self.img)
            cv2.waitKey(1)

    def calculate_and_publish_green_mass_center(self, green_mask):
            M = cv2.moments(green_mask)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                # Dibujar un círculo en el centro de masa en la imagen
                cv2.circle(self.img, (center_x, center_y), 10, (0, 255, 0), -1)
            else:
                center_x, center_y = 0, 0  # Defaults in case no red is detected

            cv2.imshow("Mascara verde", green_mask)
            cv2.imshow("<centro de masa", self.img)
            cv2.waitKey(1)

    def calculate_and_publish_yellow_mass_center(self, yellow_mask):
            M = cv2.moments(yellow_mask)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                # Dibujar un círculo en el centro de masa en la imagen
                cv2.circle(self.img, (center_x, center_y), 10, (0, 255, 0), -1)
            else:
                center_x, center_y = 0, 0  # Defaults in case no red is detected

            cv2.imshow("Mascara amarillo", yellow_mask)
            cv2.imshow("Centro de masa ", self.img)
            cv2.waitKey(1)

    def calculate_and_publish_white_mass_center(self, white_mask):
            M = cv2.moments(white_mask)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                # Dibujar un círculo en el centro de masa en la imagen
                cv2.circle(self.img, (center_x, center_y), 10, (0, 255, 0), -1)
            else:
                center_x, center_y = 0, 0  # Defaults in case no red is detected

            cv2.imshow("Mascara blanco", white_mask)
            cv2.imshow("<centro de masa ", self.img)
            cv2.waitKey(1)






def main(args=None):
    rclpy.init(args=args)
    cv_example = CVExample()
    rclpy.spin(cv_example)
    cv_example.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

