#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <grpcpp/grpcpp.h>
#include "algorithm.grpc.pb.h"

namespace drone_native {
namespace wire = ::drone::native::v1;

// Implement this class in the algorithm's own dependency environment. The SDK
// owns transport, lifecycle and request ordering; the algorithm owns its state.
class Algorithm {
 public:
  virtual ~Algorithm() = default;
  virtual wire::Capabilities Initialize(const wire::InitializeRequest&) = 0;
  virtual void Reset(const wire::ResetRequest&) = 0;
  virtual wire::Decision Step(const wire::StepRequest&) = 0;
  virtual void Close() {}
};

class Service final : public wire::Algorithm::Service {
 public:
  explicit Service(std::unique_ptr<Algorithm> algorithm);
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
  std::unique_ptr<Algorithm> algorithm_;
  wire::Capabilities capabilities_;
  std::mutex mutex_;
  std::string session_, episode_;
  uint64_t sequence_ = 0;
  double simulation_time_ = 0.0;
  bool initialized_ = false;
};

// Bind to a caller-owned Unix socket or explicit local TCP address. Returns a
// server so the executable retains ownership of signals, shutdown and logging.
std::unique_ptr<grpc::Server> StartServer(const std::string& address, Service* service);
}  // namespace drone_native
