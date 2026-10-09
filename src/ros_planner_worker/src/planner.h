#ifndef DRONE_PLAYGROUND_NATIVE_PLANNER_H_
#define DRONE_PLAYGROUND_NATIVE_PLANNER_H_

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "drone_playground/simulation/ros_planner/planner.pb.h"

namespace dp = drone_playground::ros_planner::v1;
using Cloud = std::vector<Eigen::Vector3d>;
inline Eigen::Vector3d Vector(const dp::Vec3& p) { return {p.x(), p.y(), p.z()}; }
inline void SetVector(dp::Vec3* out, const Eigen::Vector3d& p) {
  out->set_x(p.x());
  out->set_y(p.y());
  out->set_z(p.z());
}
class Planner {
 public:
  virtual ~Planner() = default;
  virtual void Map(const dp::DecideRequest& request, const Cloud& points) = 0;
  virtual bool Solve(const dp::DecideRequest& request, dp::Decision* result) = 0;
};
std::unique_ptr<Planner> MakeEgo(const dp::InitializeRequest& config);
std::unique_ptr<Planner> MakeSuper(const dp::InitializeRequest& config);
#endif
