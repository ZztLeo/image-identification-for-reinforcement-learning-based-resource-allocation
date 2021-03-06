/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/allocator.h"

#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

void AllocatorStats::Clear() {
  this->num_allocs = 0;
  this->bytes_in_use = 0;
  this->max_bytes_in_use = 0;
  this->max_alloc_size = 0;
  this->bytes_limit = 0;
}

string AllocatorStats::DebugString() const {
  return strings::Printf(
      "Limit:        %20lld\n"
      "InUse:        %20lld\n"
      "MaxInUse:     %20lld\n"
      "NumAllocs:    %20lld\n"
      "MaxAllocSize: %20lld\n",
      this->bytes_limit, this->bytes_in_use, this->max_bytes_in_use,
      this->num_allocs, this->max_alloc_size);
}

constexpr size_t Allocator::kAllocatorAlignment;

Allocator::~Allocator() {}

void RunResourceCtor(ResourceHandle* p, size_t n) {
  for (size_t i = 0; i < n; ++p, ++i) new (p) ResourceHandle();
}

void RunResourceDtor(ResourceHandle* p, size_t n) {
  for (size_t i = 0; i < n; ++p, ++i) p->~ResourceHandle();
}

// If true, cpu allocator collects more stats.
static bool cpu_allocator_collect_stats = false;
// If true, cpu allocator collects full stats.
static bool cpu_allocator_collect_full_stats = false;

// Individual allocations large than this amount will trigger a warning.
static const double kLargeAllocationWarningThreshold = 0.1;

// If cpu_allocator_collect_stats is true, warn when the total allocated memory
// exceeds this threshold.
static const double kTotalAllocationWarningThreshold = 0.5;

// Cache first invocation to port::AvailableRam, as it can be expensive.
static int64_t LargeAllocationWarningBytes() {
  static int64_t value = static_cast<int64>(port::AvailableRam() *
                                            kLargeAllocationWarningThreshold);
  return value;
}

static int64_t TotalAllocationWarningBytes() {
  static int64_t value = static_cast<int64>(port::AvailableRam() *
                                            kTotalAllocationWarningThreshold);
  return value;
}

void EnableCPUAllocatorStats(bool enable) {
  cpu_allocator_collect_stats = enable;
}
void EnableCPUAllocatorFullStats(bool enable) {
  cpu_allocator_collect_full_stats = enable;
}

class CPUAllocator : public Allocator {
 public:
  CPUAllocator() : total_allocation_warning_triggered_(false) {}

  ~CPUAllocator() override {}

  string Name() override { return "cpu"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    if (num_bytes > LargeAllocationWarningBytes()) {
      LOG(WARNING) << "Allocation of " << num_bytes << " exceeds "
                   << 100 * kLargeAllocationWarningThreshold
                   << "% of system memory.";
    }

    void* p = port::AlignedMalloc(num_bytes, alignment);
    if (cpu_allocator_collect_stats) {
      const std::size_t alloc_size = port::MallocExtension_GetAllocatedSize(p);
      mutex_lock l(mu_);
      ++stats_.num_allocs;
      stats_.bytes_in_use += alloc_size;
      stats_.max_bytes_in_use =
          std::max<int64>(stats_.max_bytes_in_use, stats_.bytes_in_use);
      stats_.max_alloc_size =
          std::max<int64>(stats_.max_alloc_size, alloc_size);

      if (stats_.bytes_in_use > TotalAllocationWarningBytes() &&
          !total_allocation_warning_triggered_) {
        LOG(WARNING) << "Total allocated memory " << stats_.bytes_in_use
                     << "exceeds " << 100 * kTotalAllocationWarningThreshold
                     << "% of system memory";
        total_allocation_warning_triggered_ = true;
      }
    }
    return p;
  }

  void DeallocateRaw(void* ptr) override {
    if (cpu_allocator_collect_stats) {
      const std::size_t alloc_size =
          port::MallocExtension_GetAllocatedSize(ptr);
      mutex_lock l(mu_);
      stats_.bytes_in_use -= alloc_size;
    }
    port::AlignedFree(ptr);
  }

  void GetStats(AllocatorStats* stats) override {
    mutex_lock l(mu_);
    *stats = stats_;
  }

  void ClearStats() override {
    mutex_lock l(mu_);
    stats_.num_allocs = 0;
    stats_.max_bytes_in_use = stats_.bytes_in_use;
    stats_.max_alloc_size = 0;
  }

  size_t AllocatedSizeSlow(const void* ptr) override {
    return port::MallocExtension_GetAllocatedSize(ptr);
  }

 private:
  mutex mu_;
  AllocatorStats stats_ GUARDED_BY(mu_);
  bool total_allocation_warning_triggered_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(CPUAllocator);
};

Allocator* cpu_allocator() {
  static Allocator* cpu_alloc = AllocatorRegistry::Global()->GetAllocator();
  if (cpu_allocator_collect_full_stats && !cpu_alloc->TracksAllocationSizes()) {
    cpu_alloc = new TrackingAllocator(cpu_alloc, true);
  }
  return cpu_alloc;
}

REGISTER_MEM_ALLOCATOR("DefaultCPUAllocator", 100, CPUAllocator);

}  // namespace tensorflow
