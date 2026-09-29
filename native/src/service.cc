#include "drone_native/algorithm.h"

#include <chrono>
#include <cmath>
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
