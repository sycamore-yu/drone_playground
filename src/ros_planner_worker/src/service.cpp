#include <grpcpp/grpcpp.h>
#include <ros/ros.h>

#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <stdexcept>

#include "drone_playground/simulation/ros_planner/planner.grpc.pb.h"
#include "planner.h"

using Clock = std::chrono::steady_clock;
double Milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void Require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
bool Finite(const dp::Vec3& p) { return Vector(p).allFinite(); }
Eigen::Quaterniond Rotation(const dp::Quaternion& p) {
  Eigen::Quaterniond q(p.w(), p.x(), p.y(), p.z());
  Require(q.coeffs().allFinite() && std::abs(q.norm() - 1.0) < 1e-4,
          "orientation must be a finite unit quaternion (w,x,y,z)");
  return q.normalized();
}
Cloud Measurements(const dp::DecideRequest& request) {
  const auto& m = request.measurement();
  Require(Finite(m.world_from_sensor().position()), "sensor position must be finite");
  Eigen::Quaterniond rotation = Rotation(m.world_from_sensor().orientation());
  Eigen::Vector3d origin = Vector(m.world_from_sensor().position());
  Cloud cloud;
  if (m.has_cloud()) {
    Require(m.cloud().points_size() <= 1000000, "point cloud too large");
    for (const auto& p : m.cloud().points()) {
      Require(Finite(p), "point cloud contains nonfinite coordinates");
      cloud.push_back(rotation * Vector(p) + origin);
    }
  } else if (m.has_depth()) {
    const auto& d = m.depth();
    Require(d.width() > 0 && d.height() > 0 && uint64_t(d.width()) * d.height() <= 1000000,
            "invalid depth dimensions");
    Require(d.metres_size() == uint64_t(d.width()) * d.height(), "depth size mismatch");
    Require(std::isfinite(d.fx()) && std::isfinite(d.fy()) && d.fx() > 0 && d.fy() > 0 &&
                std::isfinite(d.cx()) && std::isfinite(d.cy()),
            "invalid camera intrinsics");
    for (unsigned v = 0; v < d.height(); ++v) {
      for (unsigned u = 0; u < d.width(); ++u) {
        double z = d.metres(v * d.width() + u);
        Require(std::isfinite(z) && z >= 0, "depth must be finite metres; zero means no return");
        if (z == 0) continue;
        cloud.push_back(
            rotation * Eigen::Vector3d((u - d.cx()) * z / d.fx(), (v - d.cy()) * z / d.fy(), z) +
            origin);
      }
    }
  } else {
    throw std::invalid_argument("a sensor measurement is required");
  }
  return cloud;
}

class Service final : public dp::RosPlanner::Service {
 public:
  grpc::Status Initialize(grpc::ServerContext* context, const dp::InitializeRequest* request,
                          dp::Session* response) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!session_.id().empty())
      return {grpc::StatusCode::RESOURCE_EXHAUSTED, "one active session per worker"};
    try {
      Require(request->planner() == "ego" || request->planner() == "super",
              "planner must be ego or super");
      Require(std::isfinite(request->max_velocity()) && request->max_velocity() > 0 &&
                  request->max_velocity() <= 20,
              "max_velocity must be in (0,20] m/s");
      Require(std::isfinite(request->max_acceleration()) && request->max_acceleration() > 0 &&
                  request->max_acceleration() <= 30,
              "max_acceleration must be in (0,30] m/s2");
      Require(std::isfinite(request->sample_dt()) && request->sample_dt() >= 0.01 &&
                  request->sample_dt() <= 0.5,
              "sample_dt must be in [0.01,0.5] s");
      Require(std::isfinite(request->max_sensor_age()) && request->max_sensor_age() > 0 &&
                  request->max_sensor_age() <= 5,
              "max_sensor_age must be in (0,5] s");
      Require(std::isfinite(request->robot_radius()) && request->robot_radius() > 0 &&
                  request->robot_radius() <= 1,
              "robot_radius must be in (0,1] m");
      config_ = *request;
      Rebuild();
      if (context->IsCancelled()) {
        planner_.reset();
        return {grpc::StatusCode::DEADLINE_EXCEEDED, "initialization deadline expired"};
      }
      session_.set_id(std::to_string(++session_counter_));
      session_.set_episode(1);
      session_.set_planner(config_.planner());
      session_.set_upstream_commit(config_.planner() == "ego"
                                       ? "bfda51284c8c1b476043255a8145ef925a3778a5"
                                       : "2ad3419c127a617c6d7df6925e81a14175a9c096");
      *response = session_;
      return grpc::Status::OK;
    } catch (const std::exception& error) {
      return {grpc::StatusCode::INVALID_ARGUMENT, error.what()};
    }
  }
  grpc::Status Reset(grpc::ServerContext* context, const dp::Session* request,
                     dp::Session* response) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!Matches(*request))
      return {grpc::StatusCode::FAILED_PRECONDITION, "unknown session or episode"};
    try {
      Rebuild();
      if (context->IsCancelled()) {
        return {grpc::StatusCode::DEADLINE_EXCEEDED, "reset deadline expired; retry reset"};
      }
      session_.set_episode(session_.episode() + 1);
      *response = session_;
      return grpc::Status::OK;
    } catch (const std::exception& error) {
      return {grpc::StatusCode::INTERNAL, error.what()};
    }
  }
  grpc::Status Close(grpc::ServerContext*, const dp::Session* request, dp::Closed*) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!Matches(*request))
      return {grpc::StatusCode::FAILED_PRECONDITION, "unknown session or episode"};
    planner_.reset();
    session_.Clear();
    return grpc::Status::OK;
  }
  grpc::Status Decide(grpc::ServerContext* context, const dp::DecideRequest* request,
                      dp::Decision* response) override {
    const auto started = Clock::now();
    std::lock_guard<std::mutex> lock(mutex_);
    if (!Matches(request->session()) || !planner_) {
      return {grpc::StatusCode::FAILED_PRECONDITION, "initialize/reset required"};
    }
    *response->mutable_session() = session_;
    response->set_sequence(request->sequence());
    response->set_input_timestamp_ns(request->state().timestamp_ns());
    auto finish = [&](dp::PlanningStatus status, const std::string& detail) {
      response->set_status(status);
      response->set_detail(detail);
      if (status != dp::SOLVED) {
        response->clear_samples();
        response->set_valid_from_ns(0);
        response->set_valid_until_ns(0);
      }
      response->mutable_timings()->set_total_ms(Milliseconds(started));
      return grpc::Status::OK;
    };
    if (context->IsCancelled()) return finish(dp::DEADLINE_EXCEEDED, "request deadline expired");
    const auto now = request->state().timestamp_ns();
    const auto& m = request->measurement();
    if (now < 0 || now > 2147483646000000000LL || now < last_time_ ||
        request->sequence() <= sequence_ || m.timestamp_ns() < 0 ||
        m.timestamp_ns() > m.available_ns() || m.available_ns() > now ||
        now - m.timestamp_ns() > config_.max_sensor_age() * 1e9 ||
        request->goal().timestamp_ns() < 0 || request->goal().timestamp_ns() > now) {
      return finish(dp::STALE_INPUT, "timestamps/sequence violate simulation clock or sensor age");
    }
    try {
      Require(request->has_state() && request->has_goal(), "state and goal are required");
      Require(Finite(request->state().pose().position()) && Finite(request->state().velocity()) &&
                  Finite(request->state().acceleration()) && Finite(request->goal().position()) &&
                  std::isfinite(request->goal().yaw()),
              "state and goal must be finite");
      Rotation(request->state().pose().orientation());
      Cloud points = Measurements(*request);
      sequence_ = request->sequence();
      last_time_ = now;
      // A single worker owns ROS time; CPU timing uses steady_clock independently.
      ros::Time::setNow(ros::Time(now * 1e-9 + 1.0));
      dp::DecideRequest shifted = *request;
      shifted.mutable_state()->set_timestamp_ns(now + 1000000000);
      const auto mapping = Clock::now();
      planner_->Map(shifted, points);
      response->mutable_timings()->set_mapping_ms(Milliseconds(mapping));
      const auto solving = Clock::now();
      const bool solved = planner_->Solve(shifted, response);
      response->mutable_timings()->set_solve_ms(Milliseconds(solving));
      if (context->IsCancelled()) {
        planner_.reset();
        return finish(dp::DEADLINE_EXCEEDED, "solver exceeded deadline; reset required");
      }
      if (!solved)
        return finish(dp::NO_SOLUTION, "upstream planner did not produce a valid trajectory");
      response->set_valid_from_ns(response->valid_from_ns() - 1000000000);
      response->set_valid_until_ns(response->valid_until_ns() - 1000000000);
      for (auto& s : *response->mutable_samples()) {
        s.set_timestamp_ns(s.timestamp_ns() - 1000000000);
        if (!Finite(s.position()) || !Finite(s.velocity()) || !Finite(s.acceleration())) {
          return finish(dp::NO_SOLUTION, "upstream trajectory contains nonfinite values");
        }
      }
      if (response->samples_size() < 2 || response->valid_from_ns() > now + 1000 ||
          response->valid_until_ns() <= now)
        return finish(dp::NO_SOLUTION, "upstream trajectory expired");
      return finish(dp::SOLVED, "native upstream solve");
    } catch (const std::invalid_argument& error) {
      return finish(dp::INVALID_INPUT, error.what());
    } catch (const std::exception& error) {
      planner_.reset();
      return finish(dp::NO_SOLUTION, std::string(error.what()) + "; reset required");
    }
  }

 private:
  bool Matches(const dp::Session& s) const {
    return !session_.id().empty() && s.id() == session_.id() && s.episode() == session_.episode();
  }
  void Rebuild() {
    planner_.reset();
    ros::Time::setNow(ros::Time(1.0));
    planner_ = config_.planner() == "ego" ? MakeEgo(config_) : MakeSuper(config_);
    last_time_ = -1;
    sequence_ = 0;
  }
  std::mutex mutex_;
  dp::Session session_;
  dp::InitializeRequest config_;
  std::unique_ptr<Planner> planner_;
  uint64_t session_counter_{0}, sequence_{0};
  int64_t last_time_{-1};
};
int main(int argc, char** argv) {
  ros::init(argc, argv, "drone_playground_ros_planner_worker", ros::init_options::NoSigintHandler);
  ros::NodeHandle nh;
  ros::Time::setNow(ros::Time(1.0));
  Service service;
  grpc::ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(32 * 1024 * 1024);
  const std::string address = argc > 1 ? argv[1] : "0.0.0.0:50051";
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  if (!server) return 1;
  std::cout << "Native planner listening on " << address << std::endl;
  server->Wait();
}
