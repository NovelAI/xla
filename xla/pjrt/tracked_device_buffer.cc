/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/pjrt/tracked_device_buffer.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/future.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/buffer_sequencing_event.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/se_raw_buffer.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/stacktrace.h"

namespace xla {

namespace buffer_lifetime_check {
namespace {

struct Registry {
  absl::Mutex mu;
  absl::flat_hash_map<std::pair<const void*, int>, se::Stream*> streams
      ABSL_GUARDED_BY(mu);
};

Registry& GetRegistry() {
  static auto* registry = new Registry();
  return *registry;
}

std::atomic<int64_t> violations{0};

se::Stream* HomeStream(const se::DeviceAddressAllocator* allocator,
                       int device_ordinal) {
  Registry& r = GetRegistry();
  absl::MutexLock lock(r.mu);
  auto it = r.streams.find({allocator, device_ordinal});
  return it == r.streams.end() ? nullptr : it->second;
}

std::string StreamLabel(se::Stream* stream) {
  if (stream == nullptr) return "<null>";
  return absl::StrFormat("%p(%s)", stream, stream->GetName());
}

}  // namespace

Mode GetMode() {
  static const Mode mode = [] {
    const char* v = std::getenv("XLA_PJRT_BUFFER_LIFETIME_CHECK");
    if (v == nullptr) return Mode::kOff;
    absl::string_view s(v);
    if (s == "fatal") return Mode::kFatal;
    if (s == "log" || s == "1") return Mode::kLog;
    return Mode::kOff;
  }();
  return mode;
}

absl::Duration Lag() {
  static const absl::Duration lag = [] {
    const char* v = std::getenv("XLA_PJRT_BUFFER_LIFETIME_CHECK_LAG_US");
    return v == nullptr ? absl::ZeroDuration()
                        : absl::Microseconds(std::strtoll(v, nullptr, 10));
  }();
  return lag;
}

void RegisterAllocatorStream(const se::DeviceAddressAllocator* allocator,
                             int device_ordinal, se::Stream* stream) {
  Registry& r = GetRegistry();
  absl::MutexLock lock(r.mu);
  r.streams[{allocator, device_ordinal}] = stream;
}

int64_t ViolationCount() { return violations.load(); }

}  // namespace buffer_lifetime_check

ShapedBuffer RawSEDeviceMemory::AsShapedBuffer(
    PjRtDevice* device, const Shape& on_device_shape) const {
  ShapedBuffer shaped_buffer(on_device_shape, device->local_device_id().value(),
                             device->local_hardware_id().value());
  ShapeTree<se::DeviceAddressBase>::iterator iterator =
      shaped_buffer.buffers().begin();
  CHECK(iterator != shaped_buffer.buffers().end());
  iterator->second = mem();
  ++iterator;
  CHECK(iterator == shaped_buffer.buffers().end());
  return shaped_buffer;
}

class AllocatedRawSEDeviceMemory : public RawSEDeviceMemory {
 public:
  AllocatedRawSEDeviceMemory(se::DeviceAddressBase value,
                             LocalDeviceState* local_device,
                             se::DeviceAddressAllocator* allocator)
      : RawSEDeviceMemory(value),
        allocator_(allocator),
        local_device_(local_device) {
    if (local_device_->allocation_model() ==
        LocalDeviceState::kComputeSynchronized) {
      sync_point_ = local_device_->GetNextComputeStreamSyncPoint();
    }
  }

  ~AllocatedRawSEDeviceMemory() override {
    if (allocator_ &&
        buffer_lifetime_check::GetMode() != buffer_lifetime_check::Mode::kOff) {
      CheckUsesBeforeFree();
    }
    if (allocator_) {
      absl::Status status = allocator_->Deallocate(
          local_device_->local_device_id().value(), mem());
      if (!status.ok()) {
        LOG(ERROR) << "Buffer deallocation failed: " << status;
      }
    }
  }

  void UnsafeReleaseMemory() override { allocator_ = nullptr; }

  void RecordUse(se::Stream* stream, BufferSequencingEventRef event,
                 absl::string_view user) override {
    if (!event) return;
    absl::MutexLock lock(uses_mu_);
    uses_.erase(std::remove_if(uses_.begin(), uses_.end(),
                               [](const Use& u) { return u.event.IsAvailable(); }),
                uses_.end());
    uses_.push_back({stream, std::move(event), std::string(user)});
  }

  absl::StatusOr<BufferSequencingEventRef> GetDefinitionEvent(
      AsyncWorkRunner* async_work_runner, bool nullptr_if_past) const override {
    if (sync_point_ != std::numeric_limits<size_t>::max()) {
      return local_device_->GetEventForComputeStreamSyncPoint(
          sync_point_, async_work_runner, nullptr_if_past);
    }
    return BufferSequencingEventRef();
  }

 private:
  struct Use {
    se::Stream* stream;
    BufferSequencingEventRef event;
    std::string user;
  };

  void CheckUsesBeforeFree() {
    int ordinal = local_device_->local_device_id().value();
    se::Stream* home =
        buffer_lifetime_check::HomeStream(allocator_, ordinal);
    if (home == nullptr) return;
    absl::MutexLock lock(uses_mu_);
    for (const Use& u : uses_) {
      if (u.stream == home || u.event.IsAvailable()) continue;
      int64_t n = ++buffer_lifetime_check::violations;
      std::string msg = absl::StrFormat(
          "buffer lifetime violation #%d: device buffer %p (%d bytes) is being "
          "freed on its allocator's stream %s while a use on stream %s by %s "
          "has not completed and nothing orders the free after it",
          n, mem().opaque(), mem().size(),
          buffer_lifetime_check::StreamLabel(home),
          buffer_lifetime_check::StreamLabel(u.stream), u.user);
      if (buffer_lifetime_check::GetMode() ==
          buffer_lifetime_check::Mode::kFatal) {
        LOG(FATAL) << msg << "\nfree stack:\n" << tsl::CurrentStackTrace();
      }
      if (n <= 20) {
        LOG(ERROR) << msg << "\nfree stack:\n" << tsl::CurrentStackTrace();
      } else if (n % 1000 == 0) {
        LOG(ERROR) << msg << " (" << n << " violations so far)";
      }
    }
  }

  se::DeviceAddressAllocator* allocator_;
  LocalDeviceState* local_device_;
  size_t sync_point_ = std::numeric_limits<size_t>::max();
  absl::Mutex uses_mu_;
  std::vector<Use> uses_ ABSL_GUARDED_BY(uses_mu_);
};

tsl::AsyncValueRef<RawSEDeviceMemory> RawSEDeviceMemory::Create(
    se::DeviceAddressBase value, LocalDeviceState* local_device,
    se::DeviceAddressAllocator* allocator) {
  return tsl::MakeAvailableAsyncValueRef<AllocatedRawSEDeviceMemory>(
      value, local_device, allocator);
}

/*static*/ void RawSEDeviceMemory::ConstructDelayed(
    tsl::AsyncValueRef<RawSEDeviceMemory> buf, se::DeviceAddressBase value,
    LocalDeviceState* local_device, se::DeviceAddressAllocator* allocator) {
  tsl::Cast<AllocatedRawSEDeviceMemory>(buf).emplace(value, local_device,
                                                     allocator);
}

/*static*/ tsl::AsyncValueRef<RawSEDeviceMemory>
RawSEDeviceMemory::CreateDelayedMemory() {
  return tsl::MakeUnconstructedAsyncValueRef<AllocatedRawSEDeviceMemory>();
}

class ForeignRawSEDeviceMemory : public RawSEDeviceMemory {
 public:
  ForeignRawSEDeviceMemory(se::DeviceAddressBase value,
                           absl::AnyInvocable<void() &&> on_delete_callback)
      : RawSEDeviceMemory(value),
        on_delete_callback_(std::move(on_delete_callback)) {}

  ~ForeignRawSEDeviceMemory() override { std::move(on_delete_callback_)(); }

  void UnsafeReleaseMemory() override {
    LOG(FATAL) << "ForeignRawSEDeviceMemory cannot be donated.";
  }

 private:
  absl::AnyInvocable<void() &&> on_delete_callback_;
};

class SlicedRawSEDeviceMemory : public RawSEDeviceMemory {
 public:
  SlicedRawSEDeviceMemory(se::DeviceAddressBase value,
                          tsl::AsyncValueRef<RawSEDeviceMemory> base)
      : RawSEDeviceMemory(value), base_(base) {}

  void UnsafeReleaseMemory() override {
    LOG(FATAL) << "SlicedRawSEDeviceMemory cannot be donated.";
  }

  void RecordUse(se::Stream* stream, BufferSequencingEventRef event,
                 absl::string_view user) override {
    if (base_.IsAvailable()) base_->RecordUse(stream, std::move(event), user);
  }

 private:
  tsl::AsyncValueRef<RawSEDeviceMemory> base_;
};

tsl::AsyncValueRef<RawSEDeviceMemory> RawSEDeviceMemory::CreateForeign(
    se::DeviceAddressBase value,
    absl::AnyInvocable<void() &&> on_delete_callback) {
  return tsl::MakeAvailableAsyncValueRef<ForeignRawSEDeviceMemory>(
      value, std::move(on_delete_callback));
}

tsl::AsyncValueRef<RawSEDeviceMemory> RawSEDeviceMemory::CreateSlice(
    tsl::AsyncValueRef<RawSEDeviceMemory> base, size_t offset, size_t size) {
  size_t src_size = base->mem().size();
  if (offset <= src_size && size <= src_size - offset) {
    return tsl::MakeAvailableAsyncValueRef<SlicedRawSEDeviceMemory>(
        se::DeviceAddressBase(
            reinterpret_cast<char*>(base->mem().opaque()) + offset, size),
        base);
  }
  return tsl::MakeErrorAsyncValueRef(absl::InvalidArgumentError(
      absl::StrFormat("Error when slicing: [%d,%d) in array of size %d", offset,
                      offset + size, src_size)));
}



}  // namespace xla
