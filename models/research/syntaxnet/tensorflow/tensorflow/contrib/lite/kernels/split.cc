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
#include <string.h>
#include <vector>
#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace split {

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
    params = reinterpret_cast<TfLiteSplitParams*>(node->builtin_data);
    axis = GetInput(context, node, 0);
    input = GetInput(context, node, 1);
  }
  TfLiteSplitParams* params;
  TfLiteTensor* axis;
  TfLiteTensor* input;
};

TfLiteStatus UseDynamicOutputTensors(TfLiteContext* context, TfLiteNode* node) {
  for (int i = 0; i < NumOutputs(node); ++i) {
    SetTensorToDynamic(GetOutput(context, node, i));
  }
  return kTfLiteOk;
}

TfLiteStatus ResizeOutputTensors(TfLiteContext* context, TfLiteNode* node,
                                 TfLiteTensor* axis, TfLiteTensor* input,
                                 int num_splits) {
  int axis_value = GetTensorData<int>(axis)[0];
  if (axis_value < 0) {
    axis_value += NumDimensions(input);
  }

  const int input_size = SizeOfDimension(input, axis_value);
  TF_LITE_ENSURE_MSG(context, input_size % num_splits == 0,
                     "Not an even split");
  const int slice_size = input_size / num_splits;

  for (int i = 0; i < NumOutputs(node); ++i) {
    TfLiteIntArray* output_dims = TfLiteIntArrayCopy(input->dims);
    output_dims->data[axis_value] = slice_size;
    TfLiteTensor* output = GetOutput(context, node, i);
    TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_dims));
  }

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);

  OpContext op_context(context, node);

  TF_LITE_ENSURE_EQ(context, NumOutputs(node), op_context.params->num_splits);

  auto input_type = op_context.input->type;
  TF_LITE_ENSURE(context,
                 input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8);
  for (int i = 0; i < NumOutputs(node); ++i) {
    GetOutput(context, node, i)->type = input_type;
  }

  // If we know the contents of the 'axis' tensor, resize all outputs.
  // Otherwise, wait until Eval().
  if (IsConstantTensor(op_context.axis)) {
    return ResizeOutputTensors(context, node, op_context.axis, op_context.input,
                               op_context.params->num_splits);
  } else {
    return UseDynamicOutputTensors(context, node);
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);

  // When the 'axis' tensor is non-const we can't resize output tensors in
  // Prepare(), and we have to do it now.
  if (!IsConstantTensor(op_context.axis)) {
    TF_LITE_ENSURE_OK(
        context,
        ResizeOutputTensors(context, node, op_context.axis, op_context.input,
                            op_context.params->num_splits));
  }

  int axis_value = GetTensorData<int>(op_context.axis)[0];
  if (axis_value < 0) {
    axis_value += NumDimensions(op_context.input);
  }
  axis_value = RemapDim(NumDimensions(op_context.input), axis_value);

  // TODO(ahentz): Our usage of VectorOfTensors could be optimized by
  // calculating it in Prepare, unless we defer shape calculation.
  // TODO(ahentz): We can improve the optimized_ops version to handle other
  // cases too.
#define TF_LITE_SPLIT(scalar)                                                  \
  VectorOfTensors<scalar> all_outputs(*context, *node->outputs);               \
  if (axis_value == NumDimensions(op_context.input)) {                         \
    optimized_ops::TensorFlowSplit<FusedActivationFunctionType::kNone,         \
                                   scalar>(                                    \
        GetTensorData<scalar>(op_context.input),                               \
        GetTensorDims(op_context.input), NumOutputs(node), all_outputs.data(), \
        all_outputs.dims());                                                   \
  } else {                                                                     \
    reference_ops::TensorFlowSplit<scalar>(                                    \
        GetTensorData<scalar>(op_context.input),                               \
        GetTensorDims(op_context.input), axis_value, NumOutputs(node),         \
        all_outputs.data(), all_outputs.dims());                               \
  }
  switch (op_context.input->type) {
    case kTfLiteFloat32: {
      TF_LITE_SPLIT(float);
      break;
    }
    case kTfLiteUInt8: {
      TF_LITE_SPLIT(uint8_t);
      break;
    }
    default:
      context->ReportError(context,
                           "Only float32 and uint8 are currently supported.");
      return kTfLiteError;
  }
#undef TF_LITE_SPLIT

  return kTfLiteOk;
}

}  // namespace split

TfLiteRegistration* Register_SPLIT() {
  static TfLiteRegistration r = {nullptr, nullptr, split::Prepare, split::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
