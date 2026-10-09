#include <super_core/super_planner.h>
#include <yaml-cpp/yaml.h>

#include "planner.h"

// ROGMap exposes robot_state_ for synchronous embedding without ROS callback races.
class RpcMap final : public rog_map::ROGMapROS {
 public:
  using ROGMapROS::ROGMapROS;
  void State(const dp::State& state) {
    const auto& q = state.pose().orientation();
    updateRobotState(
        {Vector(state.pose().position()), Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z())});
    robot_state_.v = Vector(state.velocity());
    robot_state_.a = Vector(state.acceleration());
    robot_state_.j.setZero();
  }
};
class Super final : public Planner {
 public:
  explicit Super(const dp::InitializeRequest& config) : config_(config), nh_("~super") {
    auto yaml = YAML::LoadFile(ROS_PLANNER_CONFIG_DIR "/super.yaml");
    yaml["traj_opt"]["boundary"]["max_vel"] = config.max_velocity();
    yaml["traj_opt"]["boundary"]["max_acc"] = config.max_acceleration();
    yaml["super_planner"]["robot_r"] = config.robot_radius();
    std::ofstream out("/tmp/ros-planner-worker-super.yaml");
    out << yaml;
    out.close();
    map_ = std::make_shared<RpcMap>(nh_, "/tmp/ros-planner-worker-super.yaml");
    ros_ = std::make_shared<ros_interface::Ros1Interface>(nh_);
    planner_ =
        std::make_unique<super_planner::SuperPlanner>("/tmp/ros-planner-worker-super.yaml", ros_, map_);
  }
  void Map(const dp::DecideRequest& request, const Cloud& points) override {
    rog_map::PointCloud cloud;
    for (const auto& p : points) {
      pcl::PointXYZI point;
      point.x = p.x();
      point.y = p.y();
      point.z = p.z();
      point.intensity = 1.0;
      cloud.push_back(point);
    }
    const auto& pose = request.measurement().world_from_sensor();
    const auto& q = pose.orientation();
    map_->updateMap(cloud,
                    {Vector(pose.position()), Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z())});
    map_->State(request.state());
    rog_map::RobotState state;
    planner_->getRobotState(state);
  }
  bool Solve(const dp::DecideRequest& request, dp::Decision* result) override {
    auto goal = Vector(request.goal().position());
    if (map_->isOccupied(Vector(request.state().pose().position()))) {
      result->set_upstream_status(super_utils::FAILED);
      return false;
    }
    if (has_trajectory_) {
      auto previous = planner_->getCommittedPositionTrajectory();
      if (request.state().timestamp_ns() * 1e-9 >=
          previous.start_WT + previous.getTotalDuration()) {
        has_trajectory_ = false;
      }
    }
    bool new_goal = !has_trajectory_ || (goal - goal_).norm() > 1e-6;
    auto code = has_trajectory_ ? planner_->ReplanOnce(goal, request.goal().yaw(), new_goal)
                                : planner_->PlanFromRest(goal, request.goal().yaw(), true);
    result->set_upstream_status(static_cast<int>(code));
    if (code != super_utils::SUCCESS && code != super_utils::NO_NEED) return false;
    auto traj = planner_->getCommittedPositionTrajectory();
    if (traj.empty()) return false;
    const double start = traj.start_WT;
    const double duration = traj.getTotalDuration();
    const double now = request.state().timestamp_ns() * 1e-9;
    const double begin = std::max(0.0, now - start);
    if (begin >= duration) return false;
    has_trajectory_ = true;
    goal_ = goal;
    result->set_valid_from_ns(std::llround((start + begin) * 1e9));
    result->set_valid_until_ns(std::llround((start + duration) * 1e9));
    int count = std::max(1, static_cast<int>(std::ceil((duration - begin) / config_.sample_dt())));
    for (int i = 0; i <= count; ++i) {
      const double t = begin + (duration - begin) * i / count;
      auto* sample = result->add_samples();
      sample->set_timestamp_ns(std::llround((start + t) * 1e9));
      SetVector(sample->mutable_position(), traj.getPos(t));
      SetVector(sample->mutable_velocity(), traj.getVel(t));
      SetVector(sample->mutable_acceleration(), traj.getAcc(t));
    }
    return true;
  }

 private:
  dp::InitializeRequest config_;
  ros::NodeHandle nh_;
  std::shared_ptr<RpcMap> map_;
  ros_interface::RosInterface::Ptr ros_;
  std::unique_ptr<super_planner::SuperPlanner> planner_;
  bool has_trajectory_{false};
  Eigen::Vector3d goal_;
};
std::unique_ptr<Planner> MakeSuper(const dp::InitializeRequest& config) {
  return std::make_unique<Super>(config);
}
