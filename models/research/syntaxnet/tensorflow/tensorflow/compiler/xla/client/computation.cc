/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/computation.h"

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

Computation::Computation() : parent_(nullptr) {}

Computation::Computation(ServiceInterface* parent,
                         const ComputationHandle& handle)
    : handle_(handle), parent_(parent) {}

Computation::Computation(Computation&& computation)
    : handle_(std::move(computation.handle_)), parent_(computation.parent_) {
  computation.ResetWithoutFreeing();
}

void Computation::Reset() {
  // TODO(b/34469253) deallocate any owned computation.
  ResetWithoutFreeing();
}

StatusOr<std::unique_ptr<SessionModule>> Computation::Snapshot() const {
  SnapshotComputationRequest request;
  *request.mutable_computation() = handle_;
  SnapshotComputationResponse response;

  TF_RETURN_IF_ERROR(parent_->SnapshotComputation(&request, &response));

  return WrapUnique(response.release_module());
}

Computation::~Computation() { Reset(); }

Computation& Computation::operator=(Computation&& computation) {
  if (&computation != this) {
    Reset();
    handle_ = computation.handle_;
    parent_ = computation.parent_;
    computation.ResetWithoutFreeing();
  }
  return *this;
}

void Computation::ResetWithoutFreeing() {
  handle_.Clear();
  parent_ = nullptr;
}

StatusOr<ProgramShape> Computation::GetProgramShape() const {
  GetComputationShapeRequest request;
  *request.mutable_computation() = handle_;
  GetComputationShapeResponse response;

  TF_RETURN_IF_ERROR(parent_->GetComputationShape(&request, &response));

  return std::move(*response.mutable_program_shape());
}

}  // namespace xla
