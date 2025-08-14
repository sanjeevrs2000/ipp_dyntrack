#include <rclcpp/rclcpp.hpp>
#include <gz/transport/Node.hh>
#include <gz/msgs/pose_v.pb.h>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose.hpp>

class PoseVListener : public rclcpp::Node
{
public:
  PoseVListener(const std::string &world_name) : Node("pose_v_listener")
  {
    gz_node_.Subscribe("/world/" + world_name + "/dynamic_pose/info",
                       &PoseVListener::PoseCallback, this);

    pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/buoy_positions", 10);
                      }

private:
  void PoseCallback(const gz::msgs::Pose_V &msg)
  {
    geometry_msgs::msg::PoseArray pose_array_msg;
    pose_array_msg.header.stamp = this->now();
    pose_array_msg.header.frame_id = "map";

    for (int i = 0; i < msg.pose_size(); ++i)
    {
      const auto &pose = msg.pose(i);
      const std::string &name = pose.name();

    if (name.find("round_buoy") != std::string::npos || name.find("buoy") != std::string::npos)
      {
        auto pos = pose.position();
        // RCLCPP_INFO(this->get_logger(), "Object [%s] position: x=%.2f, y=%.2f, z=%.2f",
        //             name.c_str(), pos.x(), pos.y(), pos.z());

        geometry_msgs::msg::Pose pose_msg;
        pose_msg.position.x = pos.x();
        pose_msg.position.y = pos.y();
        pose_msg.position.z = pos.z();

        pose_array_msg.poses.push_back(pose_msg);
      }
    }
    pub_->publish(pose_array_msg);
  }

  gz::transport::Node               gz_node_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  std::string world_name = "default";

  if (argc > 1)
  {
    world_name = argv[1];
  }

  auto node = std::make_shared<PoseVListener>(world_name);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
