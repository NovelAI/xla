/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_SERVICE_GPU_GPU_AOT_COMPILATION_RESULT_H_
#define XLA_SERVICE_GPU_GPU_AOT_COMPILATION_RESULT_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/stream_executor/kernel_symbol_registry.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

// `AotCompilationResult` implementation for GPU, containing a serialized
// `GpuExecutable`.
//
// Unlike `LegacyGpuAotCompilationResult`, this result contains the entire
// optimized executable, including the Thunks, as opposed to just the optimized
// HLO.
class GpuAotCompilationResult : public AotCompilationResult {
 public:
  // The optimized HloModule is parsed lazily: the load path
  // (LoadExecutable -> GpuExecutable::FromProto) parses its own copy and never
  // asks for it, so eagerly deserializing it here doubled the HLO parse cost
  // of every executable load.
  static absl::StatusOr<std::unique_ptr<GpuAotCompilationResult>> FromProto(
      GpuExecutableProto executable) {
    return absl::WrapUnique(new GpuAotCompilationResult(std::move(executable)));
  }

  absl::StatusOr<std::string> SerializeAsString() const final {
    std::string serialized = executable_.SerializeAsString();
    if (serialized.empty()) {
      return absl::InternalError("Failed to serialize GpuExecutableProto.");
    }
    return serialized;
  }

  absl::StatusOr<std::unique_ptr<Executable>>
      LoadExecutable(const se::StreamExecutor* stream_exec) && final {
    stream_executor::Platform::Id platform_id =
        stream_exec->GetPlatform()->id();
    const auto symbol_resolver = [&](absl::string_view symbol_name) {
      stream_executor::KernelSymbolRegistry& registry =
          stream_executor::KernelSymbolRegistry::GetGlobalInstance();
      return registry.FindSymbol(symbol_name, platform_id);
    };
    return GpuExecutable::FromProto(
        executable_, stream_exec->GetDeviceDescription(),
        stream_exec->GetPlatform()->Name(), symbol_resolver);
  }

  const HloModule* optimized_module() const final {
    return EnsureHloModule().get();
  };

  std::shared_ptr<HloModule> shared_optimized_module() final {
    return EnsureHloModule();
  };

 private:
  explicit GpuAotCompilationResult(GpuExecutableProto executable)
      : executable_(std::move(executable)) {}

  const std::shared_ptr<HloModule>& EnsureHloModule() const {
    absl::MutexLock lock(hlo_module_mu_);
    if (hlo_module_ == nullptr) {
      absl::StatusOr<std::unique_ptr<HloModule>> module =
          HloModule::CreateFromProtoWithConfig(
              executable_.hlo_module_with_config());
      if (module.ok()) {
        hlo_module_ = *std::move(module);
      } else {
        LOG(ERROR) << "Failed to parse the optimized HLO module out of a "
                      "GpuExecutableProto: "
                   << module.status();
      }
    }
    return hlo_module_;
  }

  GpuExecutableProto executable_;
  mutable absl::Mutex hlo_module_mu_;
  mutable std::shared_ptr<HloModule> hlo_module_ ABSL_GUARDED_BY(hlo_module_mu_);
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_GPU_AOT_COMPILATION_RESULT_H_
