#include <sstream>
#include <string>

#include "cv_bridge/cv_bridge.hpp"
#include "image_transport/image_transport.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "sensor_msgs/msg/image.hpp"

int main(int argc, char ** argv)
{
  if (argc < 2) {
    std::cerr << "Uso: ros2 run video_publisher talker <ruta_video>\n";
    return 1;
  }

  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("video_publisher");

  image_transport::ImageTransport it(node);
  auto pub = it.advertise("camera/image", 1);

  std::string video_path = argv[1];

  cv::VideoCapture cap(video_path);

  if (!cap.isOpened()) {
    std::cerr << "No se pudo abrir el video: " << video_path << "\n";
    return 1;
  }

  double fps = cap.get(cv::CAP_PROP_FPS);
  if (fps <= 0.0) fps = 30.0;

  rclcpp::WallRate loop_rate(fps);

  cv::Mat frame;
  std_msgs::msg::Header header;

  auto logger = rclcpp::get_logger("video_publisher");
  RCLCPP_INFO(logger, "Publishing video");
  while (rclcpp::ok()) {

    cap >> frame;

    // Si llega al final del video, terminar
    if (frame.empty()){
      RCLCPP_INFO(logger, "Finished publishing video");
      break;
    }

    header.stamp = node->now();

    auto msg = cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();

    pub.publish(msg);

    rclcpp::spin_some(node);
    loop_rate.sleep();
  }

  rclcpp::shutdown();
  return 0;
}