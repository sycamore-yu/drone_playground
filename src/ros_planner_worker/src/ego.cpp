#include <pcl_conversions/pcl_conversions.h>
#include <plan_manage/planner_manager.h>

#include "planner.h"

class Ego final : public Planner {
 public:
  explicit Ego(const dp::InitializeRequest& config) : config_(config), nh_("~ego") {
    const std::vector<std::pair<std::string, double>> params = {
        {"grid_map/resolution", 0.2},
        {"grid_map/map_size_x", 30.0},
        {"grid_map/map_size_y", 30.0},
        {"grid_map/map_size_z", 12.0},
        {"grid_map/local_update_range_x", 14.0},
        {"grid_map/local_update_range_y", 14.0},
        {"grid_map/local_update_range_z", 6.0},
        {"grid_map/ground_height", -3.0},
        {"grid_map/obstacles_inflation", config.robot_radius()},
        {"grid_map/fx", 320.0},
        {"grid_map/fy", 320.0},
        {"grid_map/cx", 320.0},
        {"grid_map/cy", 240.0},
        {"grid_map/depth_filter_tolerance", 0.15},
        {"grid_map/depth_filter_maxdist", 14.0},
        {"grid_map/depth_filter_mindist", 0.1},
        {"grid_map/k_depth_scaling_factor", 1000.0},
        {"grid_map/min_ray_length", 0.1},
        {"grid_map/max_ray_length", 14.0},
        {"manager/max_vel", config.max_velocity()},
        {"manager/max_acc", config.max_acceleration()},
        {"manager/max_jerk", 20.0},
        {"manager/control_points_distance", 0.4},
        {"manager/feasibility_tolerance", 0.05},
        {"manager/planning_horizon", 7.0},
        {"optimization/lambda_smooth", 1.0},
        {"optimization/lambda_collision", 0.5},
        {"optimization/lambda_feasibility", 0.1},
        {"optimization/lambda_fitness", 1.0},
        {"optimization/dist0", 0.5},
        {"optimization/max_vel", config.max_velocity()},
        {"optimization/max_acc", config.max_acceleration()}};
    for (const auto& p : params) nh_.setParam(p.first, p.second);
    nh_.setParam("grid_map/skip_pixel", 2);
    nh_.setParam("grid_map/depth_filter_margin", 1);
    nh_.setParam("grid_map/pose_type", 1);
    auto vis = std::make_shared<ego_planner::PlanningVisualization>(nh_);
    manager_.initPlanModules(nh_, vis);
  }
  void Map(const dp::DecideRequest& request, const Cloud& points) override {
    // EGO's cloud map has no temporal fusion; recenter its finite grid each decision.
    offset_ = Vector(request.state().pose().position());
    auto odom = boost::make_shared<nav_msgs::Odometry>();
    odom->header.stamp.fromNSec(request.state().timestamp_ns());
    odom->pose.pose.orientation.w = 1.0;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    for (const auto& world : points) {
      Eigen::Vector3d p = world - offset_;
      cloud.push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
    }
    auto msg = boost::make_shared<sensor_msgs::PointCloud2>();
    pcl::toROSMsg(cloud, *msg);
    msg->header.stamp.fromNSec(request.measurement().timestamp_ns());
    msg->header.frame_id = "world";
    manager_.grid_map_->ingestCloud(odom, msg);
  }
  bool Solve(const dp::DecideRequest& request, dp::Decision* result) override {
    Eigen::Vector3d goal = Vector(request.goal().position()) - offset_;
    if (goal.norm() > 7.0) goal *= 7.0 / goal.norm();
    const double resolution = manager_.grid_map_->getResolution();
    const Eigen::Vector3d direction = goal.normalized();
    // A clipped local endpoint can lie inside an obstacle in the sensed map.
    while (goal.norm() > resolution && manager_.grid_map_->getInflateOccupancy(goal) != 0)
      goal -= resolution * direction;
    bool ok = manager_.reboundReplan(Eigen::Vector3d::Zero(), Vector(request.state().velocity()),
                                     Vector(request.state().acceleration()), goal,
                                     Eigen::Vector3d::Zero(), true, false);
    // Match the upstream FSM's recovery from a failed polynomial initialization.
    if (!ok)
      ok = manager_.reboundReplan(Eigen::Vector3d::Zero(), Vector(request.state().velocity()),
                                  Vector(request.state().acceleration()), goal,
                                  Eigen::Vector3d::Zero(), true, true);
    result->set_upstream_status(ok ? 1 : 0);
    if (!ok) return false;
    auto& traj = manager_.local_data_;
    const double duration = traj.duration_;
    const int count = std::max(1, static_cast<int>(std::ceil(duration / config_.sample_dt())));
    result->set_valid_from_ns(request.state().timestamp_ns());
    result->set_valid_until_ns(request.state().timestamp_ns() + std::llround(duration * 1e9));
    for (int i = 0; i <= count; ++i) {
      const double t = duration * i / count;
      auto* sample = result->add_samples();
      sample->set_timestamp_ns(request.state().timestamp_ns() + std::llround(t * 1e9));
      SetVector(sample->mutable_position(), traj.position_traj_.evaluateDeBoorT(t) + offset_);
      SetVector(sample->mutable_velocity(), traj.velocity_traj_.evaluateDeBoorT(t));
      SetVector(sample->mutable_acceleration(), traj.acceleration_traj_.evaluateDeBoorT(t));
    }
    return true;
  }

 private:
  dp::InitializeRequest config_;
  ros::NodeHandle nh_;
  ego_planner::EGOPlannerManager manager_;
  Eigen::Vector3d offset_;
};
std::unique_ptr<Planner> MakeEgo(const dp::InitializeRequest& config) {
  return std::make_unique<Ego>(config);
}
