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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/runtime/types.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool ResolveBatchNormalization::Run(Model* model, std::size_t op_index) {
  auto bn_it = model->operators.begin() + op_index;
  if (bn_it->get()->type != OperatorType::kBatchNormalization) {
    return false;
  }
  const auto* bn_op =
      static_cast<const BatchNormalizationOperator*>(bn_it->get());

  const auto& mean_array = model->GetArray(bn_op->inputs[1]);
  const auto& multiplier_array = model->GetArray(bn_op->inputs[2]);
  const auto& offset_array = model->GetArray(bn_op->inputs[3]);

  CHECK(IsConstantParameterArray(*model, bn_op->inputs[1]) &&
        IsConstantParameterArray(*model, bn_op->inputs[2]) &&
        IsConstantParameterArray(*model, bn_op->inputs[3]))
      << "Batch normalization resolution requires that mean, multiplier and "
         "offset arrays be constant.";

  // We should only have *float* BatchNormalizations... let's guard this
  // assumption by CHECK's.
  CHECK(mean_array.data_type == ArrayDataType::kFloat);
  CHECK(multiplier_array.data_type == ArrayDataType::kFloat);
  CHECK(offset_array.data_type == ArrayDataType::kFloat);

  // Create the new Mul, Add operators
  auto* mul_op = new MulOperator;
  auto* add_op = new AddOperator;
  const string mul_name =
      AvailableArrayName(*model, bn_op->outputs[0] + "_mul");
  const string add_name =
      AvailableArrayName(*model, bn_op->outputs[0] + "_add");
  const string mul_param_name = AvailableArrayName(*model, mul_name + "_param");
  const string add_param_name = AvailableArrayName(*model, add_name + "_param");
  mul_op->inputs = {bn_op->inputs[0], mul_param_name};
  mul_op->outputs = {mul_name};
  add_op->inputs = {mul_name, add_param_name};
  add_op->outputs = {bn_op->outputs[0]};
  AddMessageF("Splitting %s into %s and %s", LogName(*bn_op), LogName(*mul_op),
              LogName(*add_op));

  // Create the intermediate activation array (output of mul, input of add)
  auto& intermediate_array = model->GetOrCreateArray(mul_op->outputs[0]);
  intermediate_array.data_type = model->GetArray(bn_op->inputs[0]).data_type;

  // Insert the new operators in the graph
  auto add_it = model->operators.emplace(bn_it, add_op);
  auto mul_it = model->operators.emplace(add_it, mul_op);
  // update invalidated iterators.
  DCHECK_EQ(mul_it->get(), mul_op);
  add_it = mul_it + 1;
  DCHECK_EQ(add_it->get(), add_op);
  bn_it = add_it + 1;
  DCHECK_EQ(bn_it->get(), bn_op);

  // Create the new param arrays
  const auto& mean_shape = mean_array.shape();
  const auto& multiplier_shape = multiplier_array.shape();
  const auto& offset_shape = offset_array.shape();
  CHECK(mean_shape.dims() == multiplier_shape.dims());
  CHECK(mean_shape.dims() == offset_shape.dims());
  const auto& param_shape = mean_shape;
  const int buffer_size = RequiredBufferSizeForShape(param_shape);
  auto& mul_param_array = model->GetOrCreateArray(mul_param_name);
  auto& add_param_array = model->GetOrCreateArray(add_param_name);
  DropMinMax(model, mul_param_name);
  DropMinMax(model, add_param_name);
  mul_param_array.copy_shape(param_shape);
  add_param_array.copy_shape(param_shape);
  mul_param_array.data_type = ArrayDataType::kFloat;
  add_param_array.data_type = ArrayDataType::kFloat;
  auto& mul_float_data =
      mul_param_array.GetMutableBuffer<ArrayDataType::kFloat>().data;
  auto& add_float_data =
      add_param_array.GetMutableBuffer<ArrayDataType::kFloat>().data;
  mul_float_data.resize(buffer_size);
  add_float_data.resize(buffer_size);
  const auto& mean_float_data =
      mean_array.GetBuffer<ArrayDataType::kFloat>().data;
  const auto& multiplier_float_data =
      multiplier_array.GetBuffer<ArrayDataType::kFloat>().data;
  const auto& offset_float_data =
      offset_array.GetBuffer<ArrayDataType::kFloat>().data;

  CHECK(mul_float_data.size() == buffer_size);
  CHECK(add_float_data.size() == buffer_size);
  CHECK(mean_float_data.size() == buffer_size);
  CHECK(multiplier_float_data.size() == buffer_size);
  CHECK(offset_float_data.size() == buffer_size);

  for (int i = 0; i < buffer_size; i++) {
    mul_float_data[i] = multiplier_float_data[i];
    add_float_data[i] =
        offset_float_data[i] - mean_float_data[i] * multiplier_float_data[i];
  }

  // Remove the old param arrays
  model->EraseArray(bn_op->inputs[1]);
  model->EraseArray(bn_op->inputs[2]);
  model->EraseArray(bn_op->inputs[3]);

  // Remove the old operator
  DCHECK_EQ(bn_it->get(), bn_op);
  model->operators.erase(bn_it);

  return true;
}

}  // namespace toco
