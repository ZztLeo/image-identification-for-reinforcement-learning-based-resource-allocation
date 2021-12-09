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

#include "tensorflow/compiler/xla/text_literal_writer.h"

#include <string>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

/* static */ tensorflow::Status TextLiteralWriter::WriteToPath(
    const Literal& literal, tensorflow::StringPiece path) {
  std::unique_ptr<tensorflow::WritableFile> f;
  auto s = tensorflow::Env::Default()->NewWritableFile(path.ToString(), &f);
  if (!s.ok()) {
    return s;
  }

  s = f->Append(ShapeUtil::HumanString(literal.shape()) + "\n");
  if (!s.ok()) {
    return s;
  }

  tensorflow::Status status;
  tensorflow::WritableFile* f_ptr = f.get();
  literal.EachCellAsString(
      [f_ptr, &status](tensorflow::gtl::ArraySlice<int64> indices,
                       const string& value) {
        if (!status.ok()) {
          return;
        }
        string coordinates = tensorflow::strings::StrCat(
            "(", tensorflow::str_util::Join(indices, ", "), ")");

        status = f_ptr->Append(
            tensorflow::strings::StrCat(coordinates, ": ", value, "\n"));
      });
  auto ignored = f->Close();
  return status;
}

}  // namespace xla
