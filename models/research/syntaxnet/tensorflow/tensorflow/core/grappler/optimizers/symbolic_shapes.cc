/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/symbolic_shapes.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {
namespace grappler {
namespace {

BCast::Vec ShapeDims(const TensorShapeProto& shape) {
  BCast::Vec dims;
  dims.reserve(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i)
    dims.push_back(shape.dim(i).size());
  return dims;
}

}  // namespace

bool IsKnown(const TensorShapeProto::Dim& dim) { return dim.size() >= 0; }

bool IsKnownSymbolically(const TensorShapeProto::Dim& dim) {
  return dim.size() <= -2;
}

bool IsUnknown(const TensorShapeProto::Dim& dim) { return dim.size() == -1; }

bool ShapeIsSymbolicallyDefined(const TensorShapeProto& shape) {
  return !shape.unknown_rank() &&
         std::all_of(
             shape.dim().begin(), shape.dim().end(),
             [](const TensorShapeProto::Dim& dim) { return !IsUnknown(dim); });
}

bool ShapeIsSymbolicallyDefined(const OpInfo::TensorProperties& properties) {
  return ShapeIsSymbolicallyDefined(properties.shape());
}

bool ShapesSymbolicallyEqual(const TensorShapeProto& left,
                             const TensorShapeProto& right) {
  if (left.unknown_rank() || right.unknown_rank() ||
      left.dim_size() != right.dim_size()) {
    return false;
  }
  for (int i = 0; i < left.dim_size(); ++i) {
    const auto& ldim = left.dim(i);
    const auto& rdim = right.dim(i);
    if (IsUnknown(ldim) || IsUnknown(rdim) || ldim.size() != rdim.size()) {
      return false;
    }
  }
  return true;
}

bool ShapesSymbolicallyEqual(const OpInfo::TensorProperties& left,
                             const OpInfo::TensorProperties& right) {
  return ShapesSymbolicallyEqual(left.shape(), right.shape());
}

bool ShapesBroadcastable(const TensorShapeProto& left,
                         const TensorShapeProto& right) {
  if (!ShapeIsSymbolicallyDefined(left) || !ShapeIsSymbolicallyDefined(right)) {
    return false;
  }
  BCast bcast(ShapeDims(left), ShapeDims(right),
              /*fewer_dims_optimization*/ false);
  return bcast.IsValid();
}

bool ShapesBroadcastable(const OpInfo::TensorProperties& left,
                         const OpInfo::TensorProperties& right) {
  return ShapesBroadcastable(left.shape(), right.shape());
}

bool CompareSymbolicallyShapedTensorSizes(const TensorShapeProto& left,
                                          const TensorShapeProto& right) {
  // if one of the ranks is unknown, it's impossible to compare tensor sizes
  if (left.unknown_rank() || right.unknown_rank()) {
    return false;
  }

  // Tensor size, computed as a product of defined dimensions
  int64 left_defined_size = 1;
  int64 right_defined_size = 1;

  // Keep how many times each unknown dimension appeared on the left and right
  std::unordered_map<int64, int64> left_unknown_dims;
  std::unordered_map<int64, int64> right_unknown_dims;

  // Assign unique id to every unknown dimension (-1). We are going to
  // assign positive ids, because negative values are already used by
  // symbolic dimensions.
  int64 unknown_dim_id = 1;

  // For each shape dimension update "defined tensor size", if shape is defined,
  // or increment a counter for unknown dim.
  auto process_dimensions =
      [&unknown_dim_id](const TensorShapeProto& shape, int64* defined_size,
                        std::unordered_map<int64, int64>* unknown_dims) {
        for (int i = 0; i < shape.dim_size(); ++i) {
          const auto& dim = shape.dim(i);
          int64 dim_size = dim.size();
          if (dim_size > 0) {
            *defined_size *= dim_size;
          } else if (IsUnknown(dim)) {
            ++(*unknown_dims)[unknown_dim_id++];
          } else if (IsKnownSymbolically(dim)) {
            ++(*unknown_dims)[dim_size];
          }
        }
      };

  process_dimensions(left, &left_defined_size, &left_unknown_dims);
  process_dimensions(right, &right_defined_size, &right_unknown_dims);

  // Compute a union of unknown dimension ids appeared in both shapes
  std::set<int64> unknown_dims;
  for (const auto& el : left_unknown_dims) unknown_dims.insert(el.first);
  for (const auto& el : right_unknown_dims) unknown_dims.insert(el.first);

  // Cancel unknown dimensions that appeared in both shapes
  for (int64 unknown_dim : unknown_dims) {
    int64 co_occurrence = std::min(left_unknown_dims[unknown_dim],
                                   right_unknown_dims[unknown_dim]);
    left_unknown_dims[unknown_dim] -= co_occurrence;
    right_unknown_dims[unknown_dim] -= co_occurrence;
  }

  // Count unbalanced unknown dimensions
  int64 left_unbalanced_unknown_dims = 0;
  int64 right_unbalanced_unknown_dims = 0;
  for (const auto& el : left_unknown_dims)
    left_unbalanced_unknown_dims += el.second;
  for (const auto& el : right_unknown_dims)
    right_unbalanced_unknown_dims += el.second;

  if (left_unbalanced_unknown_dims == 0 && right_unbalanced_unknown_dims == 0) {
    // If unknown dimensions cancelled each other, compare tensor sizes
    // represented by defined dimensions
    return left_defined_size < right_defined_size;
  }

  if (left_defined_size <= right_defined_size &&
      left_unbalanced_unknown_dims == 0 && right_unbalanced_unknown_dims > 0) {
    // If size of a 'left" tensor computed from defined dimensions less or
    // equal, and shape on the right has unbalanced unknown dimensions, we can
    // guarantee that shape on the left is strictly smaller (assuming that
    // unknown dimension size is larger than 1)
    return true;
  }

  // In every other case, assuming that unknown dimensions can be arbitrary
  // large in size, we can't guarantee any ordering
  return false;
}

bool CompareSymbolicallyShapedTensorSizes(
    const OpInfo::TensorProperties& left,
    const OpInfo::TensorProperties& right) {
  return CompareSymbolicallyShapedTensorSizes(left.shape(), right.shape());
}

}  // end namespace grappler
}  // end namespace tensorflow
