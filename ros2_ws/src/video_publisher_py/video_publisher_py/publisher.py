import os
import sys

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class Publisher(Node):
    def __init__(self, video_path: str):
        super().__init__('video_publisher')

        if not os.path.exists(video_path):
            self.get_logger().error(f"No such file: {video_path}")
            rclpy.shutdown()
            return

        self.bridge = CvBridge()
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            self.get_logger().error(f"Can't open video: {video_path}")
            rclpy.shutdown()
            return
        
        fps = self.video.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        
        self.publisher = self.create_publisher(Image, 'camera/image', 10)

        timer_period = 1.0/fps
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.video.read()
        if not ret:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0) #Reset vide when ends
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) < 2:
        print("Usage")
        print("ros2 run video_publisher_py talker <video_path>")
        return

    node = Publisher(video_path=sys.argv[1])
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
