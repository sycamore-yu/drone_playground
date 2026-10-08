// Real cross-language protocol fixture; not a planner or a second public SDK.
#include <grpcpp/grpcpp.h>
#include "algorithm.grpc.pb.h"
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <algorithm>
#include <chrono>
#include <set>
#include <cmath>
#include <cstring>
#include <exception>
#include <stdexcept>

namespace fixture {
namespace wire = ::drone::native::v2;
class Service final : public wire::Algorithm::Service {
 public:
  grpc::Status Initialize(grpc::ServerContext*, const wire::InitializeRequest*,
                          wire::InitializeResponse*) override;
  grpc::Status Reset(grpc::ServerContext*, const wire::ResetRequest*,
                     wire::Acknowledgement*) override;
  grpc::Status Step(grpc::ServerContext*, const wire::StepRequest*,
                    wire::StepResponse*) override;
  grpc::Status Close(grpc::ServerContext*, const wire::CloseRequest*,
                     wire::Acknowledgement*) override;

 private:
  grpc::Status Check(const wire::Header&, bool reset = false);
  wire::Capabilities Configure(const wire::InitializeRequest&);
  void ResetState(const wire::ResetRequest&);
  wire::Decision Compute(const wire::StepRequest&);
  std::vector<wire::SafeFlightCorridor> Corridors(const wire::StepRequest&);
  std::string mode_;
  int count_ = 0;
  double delay_ = 0;
  wire::Capabilities capabilities_;
  std::mutex mutex_;
  std::string session_, episode_;
  uint64_t sequence_ = 0;
  double simulation_time_ = 0.0;
  bool initialized_ = false;
};


  wire::Capabilities Service::Configure(const wire::InitializeRequest& request) {
    mode_ = request.algorithm();
    auto delay = request.parameters().fields().find("delay_seconds");
    delay_ = delay == request.parameters().fields().end() ? 0.0 : delay->second.number_value();
    wire::Capabilities capabilities;
    capabilities.set_algorithm(mode_);
    capabilities.add_required_inputs("state");
    if (mode_ == "echo") {
      capabilities.add_required_inputs("upstream");
      for (auto kind : {"trajectory", "waypoint", "state", "attitude", "rates", "force_torque", "motor_rpm"}) {
        capabilities.add_accepted_upstream(kind);
        capabilities.add_outputs(kind);
      }
      capabilities.set_derivatives("none");
      return capabilities;
    }
    capabilities.add_outputs((mode_ == "no_plan" || mode_ == "visualized") ? "trajectory" : mode_);
    capabilities.set_derivatives("none");
    return capabilities;
  }
  void Service::ResetState(const wire::ResetRequest&) { count_ = 0; }
  wire::Decision Service::Compute(const wire::StepRequest& request) {
    if (delay_ > 0) std::this_thread::sleep_for(std::chrono::duration<double>(delay_));
    wire::Decision result;
    result.set_status(mode_ == "no_plan" ? wire::NO_PLAN : wire::VALID);
    result.set_plan_id(std::to_string(++count_));
    result.set_generated_at(request.header().simulation_time());
    result.set_valid_until(request.header().simulation_time() + 2.0);
    if (mode_ == "echo") {
      if (request.has_trajectory()) {
        *result.mutable_trajectory() = request.trajectory();
        double end = request.trajectory().start_time();
        for (const auto& segment : request.trajectory().segments()) end += segment.duration();
        result.set_valid_until(end);
      } else if (request.has_waypoint()) {
        *result.mutable_waypoint() = request.waypoint();
      } else if (request.has_state_setpoint()) {
        *result.mutable_state_setpoint() = request.state_setpoint();
      } else if (request.has_attitude()) {
        *result.mutable_attitude() = request.attitude();
      } else if (request.has_rates()) {
        *result.mutable_rates() = request.rates();
      } else if (request.has_force_torque()) {
        *result.mutable_force_torque() = request.force_torque();
      } else if (request.has_motor_rpm()) {
        *result.mutable_motor_rpm() = request.motor_rpm();
      } else {
        result.set_status(wire::NO_PLAN);
      }
    } else if (mode_ == "trajectory" || mode_ == "visualized") {
      auto* trajectory = result.mutable_trajectory();
      trajectory->set_start_time(request.header().simulation_time());
      trajectory->set_frame("world");
      auto* segment = trajectory->add_segments();
      segment->set_duration(2.0);
      segment->set_coefficient_count(2);
      for (double value : {request.state().position().x(), 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0})
        segment->add_coefficients(value);
    } else if (mode_ == "waypoint") {
      auto* waypoint = result.mutable_waypoint();
      auto* point = waypoint->add_positions();
      point->set_x(request.state().position().x() + 3.0);
      point->set_z(1.0);
      waypoint->set_tolerance(0.5);
      waypoint->set_generated_at(request.header().simulation_time());
      waypoint->set_valid_until(request.header().simulation_time() + 2.0);
    } else if (mode_ == "attitude") {
      auto* command = result.mutable_attitude();
      command->mutable_rpy();
      command->set_thrust(0.35);
    } else if (mode_ == "rates") {
      auto* command = result.mutable_rates();
      command->mutable_body_rates();
      command->set_thrust(0.35);
    } else if (mode_ == "state") {
      auto* command = result.mutable_state_setpoint();
      auto* velocity = command->mutable_velocity();
      velocity->set_x(1.0); velocity->set_y(2.0); velocity->set_z(3.0);
      command->set_yaw(0.0);
    }
    return result;
  }
  std::vector<wire::SafeFlightCorridor> Service::Corridors(const wire::StepRequest& request) {
    if (mode_ != "visualized") return {};
    wire::SafeFlightCorridor corridor;
    corridor.set_name("candidate");
    corridor.set_frame("world");
    corridor.set_generated_at(request.header().simulation_time());
    corridor.set_valid_until(request.header().simulation_time() + 2.0);
    auto* poly = corridor.add_polytopes();
    for (double value : {1.,0.,0.,-1., -1.,0.,0.,-1., 0.,1.,0.,-1.,
                         0.,-1.,0.,-1., 0.,0.,1.,-1., 0.,0.,-1.,-1.})
      poly->add_halfspaces(value);
    return {corridor};
  }

namespace {
grpc::Status Invalid(const std::string& message) {
  return {grpc::StatusCode::INVALID_ARGUMENT, message};
}
grpc::Status Failed(const std::string& message) {
  return {grpc::StatusCode::FAILED_PRECONDITION, message};
}
bool Finite(const wire::Vec3& v) {
  return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}
void Require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
void Validate(const wire::Waypoint& path) {
  Require(!path.positions().empty() && std::isfinite(path.tolerance()) && path.tolerance() > 0,
          "Waypoints require positions and positive tolerance");
  for (const auto& p : path.positions()) Require(Finite(p), "Nonfinite waypoint");
}
// An upstream goal outside its own declared window is stale input, not a usable
// setpoint. A zero valid_until means the producer declared no horizon.
void ValidateTimed(const wire::Waypoint& path, double now) {
  Validate(path);
  Require(std::isfinite(path.generated_at()) && path.generated_at() >= 0 &&
          path.generated_at() <= now + 1e-9,
          "Upstream waypoint has a future generation time");
  Require(std::isfinite(path.valid_until()) && path.valid_until() >= 0 &&
          (path.valid_until() == 0 || path.valid_until() >= now),
          "Upstream waypoint is expired");
}
template <class Output>
std::string ValidateControl(const Output& output) {
  if (output.has_state_setpoint()) {
    const auto& value = output.state_setpoint();
    Require(value.has_position() || value.has_velocity() || value.has_acceleration() ||
            value.has_yaw() || value.has_yaw_rate(), "State setpoint requires a controlled field");
    Require((!value.has_position() || Finite(value.position())) &&
            (!value.has_velocity() || Finite(value.velocity())) &&
            (!value.has_acceleration() || Finite(value.acceleration())) &&
            (!value.has_yaw() || std::isfinite(value.yaw())) &&
            (!value.has_yaw_rate() || std::isfinite(value.yaw_rate())),
            "Nonfinite state setpoint");
    return "state";
  }
  if (output.has_attitude()) {
    const auto& value = output.attitude();
    Require(value.has_rpy() && Finite(value.rpy()) && std::isfinite(value.thrust()),
            "Attitude requires finite rpy and collective thrust");
    return "attitude";
  }
  if (output.has_rates()) {
    const auto& value = output.rates();
    Require(value.has_body_rates() && Finite(value.body_rates()) && std::isfinite(value.thrust()),
            "Rates require finite body_rates and collective thrust");
    return "rates";
  }
  if (output.has_force_torque()) {
    const auto& value = output.force_torque();
    Require(value.has_torque() && Finite(value.torque()) && std::isfinite(value.thrust()),
            "Force/torque requires finite body torque and collective thrust");
    return "force_torque";
  }
  if (output.has_motor_rpm()) {
    const auto& value = output.motor_rpm();
    Require(value.rpm_size() == 4, "MotorRPM requires four rotor speeds");
    for (double speed : value.rpm()) Require(std::isfinite(speed), "Nonfinite motor speed");
    return "motor_rpm";
  }
  return {};
}
double Validate(const wire::Trajectory& curve, double now) {
  double end = curve.start_time();
  Require(curve.frame() == "world",
          "Trajectory coefficients must use the world frame");
  Require(std::isfinite(end) && end >= 0 && end <= now + 1e-9 && !curve.segments().empty(),
          "Trajectory must cover the current time");
  for (const auto& segment : curve.segments()) {
    Require(std::isfinite(segment.duration()) && segment.duration() > 0,
            "Trajectory duration must be positive and finite");
    Require(segment.coefficient_count() >= 1 && segment.coefficient_count() <= 16 &&
            segment.coefficients_size() == 4 * segment.coefficient_count(),
            "Invalid trajectory coefficient shape");
    for (double v : segment.coefficients()) Require(std::isfinite(v), "Nonfinite coefficient");
    end += segment.duration();
  }
  Require(std::isfinite(end) && now <= end + 1e-9, "Trajectory is expired");
  return end;
}
void ValidateBuffer(const std::string& bytes, uint64_t count, bool depth) {
  Require(bytes.size() % 4 == 0 && count == bytes.size() / 4, "Measurement shape/bytes mismatch");
  for (size_t i = 0; i < bytes.size(); i += 4) {
    uint32_t bits = 0;
    for (unsigned j = 0; j < 4; ++j)
      bits |= static_cast<uint32_t>(static_cast<unsigned char>(bytes[i+j])) << (8*j);
    float value;
    static_assert(sizeof(value) == 4, "Protocol requires float32");
    std::memcpy(&value, &bits, 4);
    Require(std::isfinite(value) && (!depth || value >= 0), "Invalid measurement value");
  }
}
void ValidateInputs(const wire::StepRequest& request, const wire::Capabilities& capabilities) {
  std::set<std::string> available{"state"};
  if (request.state().has_angular_velocity()) {
    Require(Finite(request.state().angular_velocity()), "Nonfinite body angular velocity");
    available.insert("angular_velocity");
  }
  if (request.has_goal()) { Validate(request.goal()); available.insert("goal"); }
  if (request.has_measurement()) {
    const auto& measurement = request.measurement();
    Require(std::isfinite(measurement.time()) && measurement.time() >= 0 &&
            measurement.time() <= request.header().simulation_time() + 1e-9,
            "Invalid measurement timestamp");
    if (measurement.has_depth()) {
      const auto& depth = measurement.depth();
      Require(measurement.frame() == "camera_optical" && depth.height() && depth.width(),
              "Invalid depth frame/shape");
      Require(Finite(depth.camera_position()) && depth.camera_quaternion_xyzw_size() == 4,
              "Invalid depth camera pose");
      double norm = 0;
      for (double q : depth.camera_quaternion_xyzw()) norm += q*q;
      Require(std::isfinite(norm) && std::abs(std::sqrt(norm) - 1) <= 1e-4,
              "Depth camera requires unit xyzw quaternion");
      ValidateBuffer(depth.float32_le(), uint64_t(depth.height()) * depth.width(), true);
      available.insert("depth");
    } else if (measurement.has_point_cloud()) {
      const auto& cloud = measurement.point_cloud();
      Require((measurement.frame() == "world" || measurement.frame() == "body") &&
              (cloud.channels() == 3 || cloud.channels() == 4), "Invalid point cloud frame/shape");
      ValidateBuffer(cloud.float32_le(), uint64_t(cloud.count()) * cloud.channels(), false);
      available.insert("point_cloud");
    } else throw std::invalid_argument("Missing measurement payload");
  }
  std::string kind;
  if (request.has_trajectory()) {
    Validate(request.trajectory(), request.header().simulation_time());
    kind = "trajectory"; available.insert("trajectory");
  } else if (request.has_waypoint()) {
    ValidateTimed(request.waypoint(), request.header().simulation_time());
    kind = "waypoint"; available.insert("waypoint");
  } else {
    kind = ValidateControl(request);
    if (!kind.empty()) available.insert(kind == "state" ? "state_setpoint" : kind);
  }
  std::set<std::string> accepted(capabilities.accepted_upstream().begin(),
                                 capabilities.accepted_upstream().end());
  const std::set<std::string> required(capabilities.required_inputs().begin(),
                                      capabilities.required_inputs().end());
  if (!kind.empty()) {
    Require(accepted.count(kind), "Unsupported upstream physical interface");
    available.insert("upstream");
  }
  for (const auto& input : required)
    Require(available.count(input), "Missing required native algorithm input");
}

void ValidateDecision(const wire::Decision& decision, double now,
                      const wire::Capabilities& capabilities) {
  const auto status = decision.status();
  if (status != wire::VALID) {
    Require(status == wire::NO_PLAN || status == wire::INFEASIBLE ||
                status == wire::BUDGET_EXHAUSTED,
            "Unsupported decision status");
    Require(decision.output_case() == wire::Decision::OUTPUT_NOT_SET,
            "Unsuccessful decision cannot carry executable output");
    return;
  }

  Require(!decision.plan_id().empty(), "Valid decision requires a plan identity");
  Require(std::isfinite(decision.generated_at()) && decision.generated_at() >= 0 &&
              decision.generated_at() <= now + 1e-9 &&
              std::isfinite(decision.valid_until()) && decision.valid_until() >= now,
          "Decision has a future origin or expired validity");

  std::string kind;
  if (decision.has_trajectory()) {
    const double end = Validate(decision.trajectory(), now);
    Require(decision.valid_until() <= end + 1e-9,
            "Trajectory does not cover its declared decision validity");
    kind = "trajectory";
  } else if (decision.has_waypoint()) {
    Validate(decision.waypoint());
    kind = "waypoint";
  } else {
    kind = ValidateControl(decision);
    Require(!kind.empty(), "Valid decision requires an executable output");
  }

  const std::set<std::string> outputs(capabilities.outputs().begin(), capabilities.outputs().end());
  Require(outputs.count(kind), "Decision output violates advertised capabilities");
}
}  // namespace


grpc::Status Service::Check(const wire::Header& header, bool reset) {
  if (!initialized_ || header.session_id() != session_)
    return Failed("Initialize a matching session first");
  if (header.protocol_version() != 2 || header.sequence() <= sequence_)
    return Invalid("Unsupported protocol or stale request sequence");
  if (!std::isfinite(header.simulation_time()) || header.simulation_time() < 0)
    return Invalid("Simulation time must be finite and nonnegative");
  if (reset) {
    if (header.episode_id().empty() || header.episode_id() == episode_)
      return Invalid("Reset requires a fresh episode identity");
  } else {
    if (episode_.empty() || header.episode_id() != episode_)
      return Failed("Reset a matching episode first");
    if (header.simulation_time() < simulation_time_)
      return Invalid("Simulation time moved backwards");
  }
  sequence_ = header.sequence();
  return grpc::Status::OK;
}

grpc::Status Service::Initialize(grpc::ServerContext*, const wire::InitializeRequest* request,
                                 wire::InitializeResponse* response) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) return Failed("Session already initialized; close it first");
  const auto& header = request->header();
  if (header.protocol_version() != 2 || header.session_id().empty() || header.sequence() == 0)
    return Invalid("Initialize requires version 2 and a session/request identity");
  try {
    auto capabilities = Configure(*request);
    if (capabilities.algorithm() != request->algorithm() || capabilities.outputs().empty())
      return Invalid("Algorithm capability mismatch");
    capabilities_ = capabilities;
    *response->mutable_capabilities() = capabilities;
    *response->mutable_header() = header;
    (*response->mutable_provenance())["service"] = "drone-native-cpp-v2";
    session_ = header.session_id();
    sequence_ = header.sequence();
    episode_.clear();
    initialized_ = true;
    return grpc::Status::OK;
  } catch (const std::exception& error) { return Invalid(error.what()); }
}

grpc::Status Service::Reset(grpc::ServerContext*, const wire::ResetRequest* request,
                            wire::Acknowledgement* response) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto status = Check(request->header(), true);
  if (!status.ok()) return status;
  try {
    ResetState(*request);
    episode_ = request->header().episode_id();
    simulation_time_ = request->header().simulation_time();
    *response->mutable_header() = request->header();
    return grpc::Status::OK;
  } catch (const std::exception& error) {
    episode_.clear();
    return {grpc::StatusCode::INTERNAL, error.what()};
  }
}

grpc::Status Service::Step(grpc::ServerContext* context, const wire::StepRequest* request,
                           wire::StepResponse* response) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto status = Check(request->header());
  if (!status.ok()) return status;
  if (!std::isfinite(request->solve_budget_seconds()) || request->solve_budget_seconds() <= 0)
    return Invalid("Solve budget must be finite and positive");
  const auto& body = request->state();
  auto finite = [](const wire::Vec3& v) {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
  };
  if (!request->has_state() || !body.has_position() || !body.has_velocity() ||
      !finite(body.position()) || !finite(body.velocity()) || body.quaternion_xyzw_size() != 4)
    return Invalid("A finite physical body state is required");
  double norm = 0;
  for (const double q : body.quaternion_xyzw()) norm += q * q;
  if (!std::isfinite(norm) || std::abs(std::sqrt(norm) - 1) > 1e-4)
    return Invalid("Body orientation must be a unit xyzw quaternion");
  try { ValidateInputs(*request, capabilities_); }
  catch (const std::invalid_argument& error) { return Invalid(error.what()); }
  if (context->IsCancelled()) return {grpc::StatusCode::CANCELLED, "Request cancelled"};
  try {
    const auto start = std::chrono::steady_clock::now();
    *response->mutable_decision() = Compute(*request);
    const double now = request->header().simulation_time();
    for (auto& corridor : Corridors(*request)) {
      Require(corridor.frame() == "world" && std::isfinite(corridor.generated_at()) &&
              std::isfinite(corridor.valid_until()) && corridor.generated_at() >= 0 &&
              corridor.generated_at() <= now + 1e-9 && corridor.valid_until() >= corridor.generated_at(),
              "Invalid corridor frame or validity");
      *response->add_corridors() = std::move(corridor);
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
    (*response->mutable_diagnostics())["algorithm_seconds"] = elapsed.count();
    if (elapsed.count() > request->solve_budget_seconds()) {
      response->mutable_decision()->Clear();
      response->clear_corridors();
      response->clear_trajectory_previews();
      response->mutable_decision()->set_status(wire::BUDGET_EXHAUSTED);
      response->mutable_decision()->set_explanation("Algorithm exceeded its wall-clock solve budget");
    }
    ValidateDecision(response->decision(), request->header().simulation_time(), capabilities_);
    *response->mutable_header() = request->header();
    simulation_time_ = request->header().simulation_time();
    if (context->IsCancelled()) {
      episode_.clear();
      return {grpc::StatusCode::CANCELLED, "Decision completed after cancellation; reset required"};
    }
    return grpc::Status::OK;
  } catch (const std::exception& error) {
    episode_.clear();
    return {grpc::StatusCode::INTERNAL, error.what()};
  }
}

grpc::Status Service::Close(grpc::ServerContext*, const wire::CloseRequest* request,
                            wire::Acknowledgement* response) {
  std::lock_guard<std::mutex> lock(mutex_);
  // Close must remain available after initialization/reset/step failures.
  if (!initialized_ || request->header().session_id() != session_)
    return Failed("Cannot close another session");
  if (request->header().protocol_version() != 2 || request->header().sequence() <= sequence_)
    return Invalid("Unsupported protocol or stale close request");
  *response->mutable_header() = request->header();
  initialized_ = false;
  episode_.clear();
  session_.clear();
  return grpc::Status::OK;
}

std::unique_ptr<grpc::Server> StartServer(const std::string& address, Service* service) {
  grpc::ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(64 * 1024 * 1024);
  builder.SetMaxSendMessageSize(64 * 1024 * 1024);
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(service);
  auto server = builder.BuildAndStart();
  if (!server) throw std::runtime_error("Cannot bind native service address");
  return server;
}
}  // namespace fixture

int main(int argc, char** argv) {
  if (argc != 2) { std::cerr << "usage: interop_server ADDRESS\n"; return 2; }
  fixture::Service service;
  auto server = fixture::StartServer(argv[1], &service);
  server->Wait();
}
