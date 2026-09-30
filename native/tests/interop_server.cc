// Protocol fixture only: these analytic outputs are not a planning benchmark.
#include "drone_native/algorithm.h"
#include <chrono>
#include <iostream>
#include <thread>

namespace wire = drone_native::wire;
class Fixture final : public drone_native::Algorithm {
 public:
  wire::Capabilities Initialize(const wire::InitializeRequest& request) override {
    mode_ = request.algorithm();
    auto delay = request.parameters().fields().find("delay_seconds");
    delay_ = delay == request.parameters().fields().end() ? 0.0 : delay->second.number_value();
    wire::Capabilities capabilities;
    capabilities.set_algorithm(mode_);
    capabilities.add_required_inputs("state");
    if (mode_ == "echo") {
      capabilities.add_required_inputs("upstream");
      for (auto kind : {"trajectory", "waypoint", "attitude_thrust", "velocity_yaw", "world_acceleration"}) {
        capabilities.add_accepted_upstream(kind);
        capabilities.add_outputs(kind);
      }
      capabilities.set_derivatives("none");
      return capabilities;
    }
    capabilities.add_outputs(mode_ == "motion" ? "velocity_yaw" :
                             mode_ == "attitude" ? "attitude_thrust" :
                             (mode_ == "no_plan" || mode_ == "visualized") ? "trajectory" : mode_);
    capabilities.set_derivatives("none");
    return capabilities;
  }
  void Reset(const wire::ResetRequest&) override { count_ = 0; }
  wire::Decision Step(const wire::StepRequest& request) override {
    if (delay_ > 0) std::this_thread::sleep_for(std::chrono::duration<double>(delay_));
    wire::Decision result;
    result.set_status(mode_ == "no_plan" ? wire::NO_PLAN : wire::VALID);
    result.set_plan_id(std::to_string(++count_));
    result.set_generated_at(request.header().simulation_time());
    result.set_valid_until(request.header().simulation_time() + 2.0);
    if (mode_ == "echo") {
      if (request.has_reference()) {
        *result.mutable_trajectory() = request.reference();
        double end = request.reference().start_time();
        for (const auto& segment : request.reference().segments()) end += segment.duration();
        result.set_valid_until(end);
      } else if (request.has_waypoints()) {
        *result.mutable_waypoint() = request.waypoints();
      } else if (request.has_motion_command()) {
        *result.mutable_motion_command() = request.motion_command();
      } else {
        result.set_status(wire::NO_PLAN);
      }
    } else if (mode_ == "trajectory" || mode_ == "visualized") {
      auto* trajectory = result.mutable_trajectory();
      trajectory->set_start_time(request.header().simulation_time());
      auto* segment = trajectory->add_segments();
      segment->set_duration(2.0);
      segment->set_coefficient_count(2);
      for (double value : {request.state().position().x(), 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0})
        segment->add_coefficients(value);
    } else if (mode_ == "waypoint") {
      auto* point = result.mutable_waypoint()->add_positions();
      point->set_x(request.state().position().x() + 3.0);
      point->set_z(1.0);
      result.mutable_waypoint()->set_tolerance(0.5);
    } else if (mode_ == "attitude") {
      auto* command = result.mutable_motion_command();
      command->set_kind("attitude_thrust");
      for (double value : {0.0, 0.0, 0.0, 0.35}) command->add_values(value);
    } else if (mode_ == "motion") {
      auto* command = result.mutable_motion_command();
      command->set_kind("velocity_yaw");
      for (double value : {1.0, 2.0, 3.0, 0.0}) command->add_values(value);
    }
    return result;
  }
  wire::PlannerGeometry Geometry(const wire::StepRequest& request) override {
    wire::PlannerGeometry result;
    if (mode_ != "visualized") return result;
    result.set_frame("world");
    result.set_generated_at(request.header().simulation_time());
    result.set_valid_until(request.header().simulation_time() + 2.0);
    auto* corridor = result.add_corridors();
    corridor->set_name("candidate");
    auto* poly = corridor->add_polytopes();
    for (double value : {1.,0.,0.,-1., -1.,0.,0.,-1., 0.,1.,0.,-1.,
                         0.,-1.,0.,-1., 0.,0.,1.,-1., 0.,0.,-1.,-1.})
      poly->add_halfspaces(value);
    return result;
  }
 private:
  std::string mode_;
  int count_ = 0;
  double delay_ = 0;
};

int main(int argc, char** argv) {
  if (argc != 2) { std::cerr << "usage: interop_server ADDRESS\n"; return 2; }
  drone_native::Service service(std::make_unique<Fixture>());
  auto server = drone_native::StartServer(argv[1], &service);
  server->Wait();
}
