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
        self.lower = np.array([136, 87, 111], np.uint8) 
        self.upper = np.array([180, 255, 255], np.uint8) 

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
               red_mask = cv2.inRange(hsvFrame, self.lower, self.upper)
               kernel = np.ones((5, 5), "uint8") 
               red_mask = cv2.dilate(red_mask, kernel) 
               
               res_red = cv2.bitwise_and(self.img, self.img, mask = red_mask)
               gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
               gray[red_mask == 0] = 128
 
               contours, hierarchy = cv2.findContours(red_mask, 
                                                      cv2.RETR_TREE, 
                                                      cv2.CHAIN_APPROX_SIMPLE) 
               for pic, contour in enumerate(contours):
                   area = cv2.contourArea(contour) 
                   if(area > 300): 
                       x, y, w, h = cv2.boundingRect(contour) 
                       self.img = cv2.rectangle(self.img, (x, y),(x + w, y + h),  
                                       (0, 0, 255), 2) 
                       cv2.putText(self.img, "Red Colour", (x, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                                   (0, 0, 255))  
                       
              # bgr_image = cv2.cvtColor(hsvFrame, cv2.COLOR_HSV2BGR)

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
