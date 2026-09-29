#include "drone_native/algorithm.h"

#include <algorithm>
#include <chrono>
#include <set>
#include <cmath>
#include <cstring>
#include <exception>
#include <stdexcept>

namespace drone_native {
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
void Validate(const wire::MotionCommand& command) {
  const auto& kind = command.kind();
  const bool four = kind == "attitude_thrust" || kind == "thrust_bodyrates" ||
                    kind == "motor_rpm" || kind == "velocity_yaw";
  Require(four || kind == "world_acceleration", "Unknown motion command");
  Require(command.values_size() == (four ? 4 : 3), "Incorrect motion command width");
  for (double v : command.values()) Require(std::isfinite(v), "Nonfinite motion command");
}
double Validate(const wire::Trajectory& curve, double now) {
  double end = curve.start_time();
  Require(std::isfinite(end) && end <= now + 1e-9 && !curve.segments().empty(),
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
  if (request.has_reference()) {
    Validate(request.reference(), request.header().simulation_time());
    kind = "trajectory"; available.insert("reference");
  } else if (request.has_waypoints()) {
    Validate(request.waypoints()); kind = "waypoint"; available.insert("waypoints");
  } else if (request.has_motion_command()) {
    Validate(request.motion_command()); kind = request.motion_command().kind();
    available.insert("motion_command");
  }
  std::set<std::string> accepted(capabilities.accepted_upstream().begin(),
                                 capabilities.accepted_upstream().end());
  const std::set<std::string> required(capabilities.required_inputs().begin(),
                                      capabilities.required_inputs().end());
  if (accepted.empty() && required.count("reference")) accepted.insert("trajectory");
  if (!kind.empty()) {
    Require(accepted.count(kind), "Unsupported upstream physical interface");
    available.insert("upstream");
  }
  for (const auto& input : required)
    Require(available.count(input), "Missing required native algorithm input");
}
}  // namespace

Service::Service(std::unique_ptr<Algorithm> algorithm) : algorithm_(std::move(algorithm)) {}

grpc::Status Service::Check(const wire::Header& header, bool reset) {
  if (!initialized_ || header.session_id() != session_)
    return Failed("Initialize a matching session first");
  if (header.protocol_version() != 1 || header.sequence() <= sequence_)
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
  if (header.protocol_version() != 1 || header.session_id().empty() || header.sequence() == 0)
    return Invalid("Initialize requires version 1 and a session/request identity");
  try {
    auto capabilities = algorithm_->Initialize(*request);
    if (capabilities.algorithm() != request->algorithm() || capabilities.outputs().empty())
      return Invalid("Algorithm capability mismatch");
    capabilities_ = capabilities;
    *response->mutable_capabilities() = capabilities;
    *response->mutable_header() = header;
    (*response->mutable_provenance())["service"] = "drone-native-cpp-v1";
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
    algorithm_->Reset(*request);
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
    *response->mutable_decision() = algorithm_->Step(*request);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
    (*response->mutable_diagnostics())["algorithm_seconds"] = elapsed.count();
    if (elapsed.count() > request->solve_budget_seconds()) {
      response->mutable_decision()->Clear();
      response->mutable_decision()->set_status(wire::BUDGET_EXHAUSTED);
      response->mutable_decision()->set_explanation("Algorithm exceeded its wall-clock solve budget");
    }
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
  if (request->header().protocol_version() != 1 || request->header().sequence() <= sequence_)
    return Invalid("Unsupported protocol or stale close request");
  try { algorithm_->Close(); }
  catch (const std::exception& error) {
    initialized_ = false;
    return {grpc::StatusCode::INTERNAL, error.what()};
  }
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
}  // namespace drone_native
