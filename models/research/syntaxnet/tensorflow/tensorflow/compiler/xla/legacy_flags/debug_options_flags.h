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

#ifndef TENSORFLOW_COMPILER_XLA_LEGACY_FLAGS_DEBUG_OPTIONS_FLAGS_H_
#define TENSORFLOW_COMPILER_XLA_LEGACY_FLAGS_DEBUG_OPTIONS_FLAGS_H_

#include <vector>

#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {
namespace legacy_flags {

// Appends flag definitions for debug options to flag_list.
void AppendDebugOptionsFlags(std::vector<tensorflow::Flag>* flag_list);

// Fetches a DebugOptions proto message from flags provided to the program.
// Flags must be registered with the flags parser using AppendDebugOptionsFlags
// first.
xla::DebugOptions GetDebugOptionsFromFlags();

}  // namespace legacy_flags
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_LEGACY_FLAGS_DEBUG_OPTIONS_FLAGS_H_
