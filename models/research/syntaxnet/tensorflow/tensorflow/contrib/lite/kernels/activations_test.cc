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
#include <cstdarg>
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseActivationsOpModel : public SingleOpModel {
 public:
  // Most activations don't take any options, so this constructor works for
  // them.
  BaseActivationsOpModel(BuiltinOperator type, TensorData input) {
    input_ = AddInput(input);
    if (input.type == TensorType_UINT8) {
      output_ = AddOutput({input.type, {}, 0, 0, 1. / 256});
    } else {
      output_ = AddOutput({input.type, {}});
    }
    SetBuiltinOp(type, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_)});
  }

  // A dedicated constructor for SOFTMAX, which does some options.
  BaseActivationsOpModel(float softmax_beta, TensorData input) {
    input_ = AddInput(input);
    if (input.type == TensorType_UINT8) {
      output_ = AddOutput({input.type, {}, 0, 0, 1. / 256});
    } else {
      output_ = AddOutput({input.type, {}});
    }
    SetBuiltinOp(BuiltinOperator_SOFTMAX, BuiltinOptions_SoftmaxOptions,
                 CreateSoftmaxOptions(builder_, softmax_beta).Union());
    BuildInterpreter({GetShape(input_)});
  }

  BaseActivationsOpModel(BuiltinOperator type, const TensorData &input,
                         const TensorData &output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(type, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_)});
  }

 protected:
  int input_;
  int output_;
};

class FloatActivationsOpModel : public BaseActivationsOpModel {
 public:
  using BaseActivationsOpModel::BaseActivationsOpModel;

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

// TODO(ahentz): I don't quite understand the tradeoffs in the quantized
// implementation of sigmoid and software, but a tolerance of twice the output
// scale seems reasonable. We might want to change this if we have a better
// theoretical bound.
const float kQuantizedTolerance = 2 * (1. / 256);

class QuantizedActivationsOpModel : public BaseActivationsOpModel {
 public:
  using BaseActivationsOpModel::BaseActivationsOpModel;

  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(input_, data);
  }
  std::vector<uint8_t> GetOutput() { return ExtractVector<uint8_t>(output_); }
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

TEST(FloatActivationsOpTest, Relu) {
  FloatActivationsOpModel m(BuiltinOperator_RELU,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 2, 4,   //
                                 3, 0, 10, 1,  //
                             }));
}

TEST(FloatActivationsOpTest, Relu1) {
  FloatActivationsOpModel m(BuiltinOperator_RELU_N1_TO_1,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0.0, -0.6, 0.2, -0.4,  //
      0.3, -2.0, 1.1, -0.1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0.0, -0.6, 0.2, -0.4,  //
                                 0.3, -1.0, 1.0, -0.1,  //
                             }));
}

TEST(FloatActivationsOpTest, Relu6) {
  FloatActivationsOpModel m(BuiltinOperator_RELU6,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 2, 4,  //
                                 3, 0, 6, 1,  //
                             }));
}

TEST(FloatActivationsOpTest, Tanh) {
  FloatActivationsOpModel m(BuiltinOperator_TANH,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 0, -0.9999877, 0.9640275, 0.999329,    //
                                 0.99505475, -0.9640275, 1, 0.7615941,  //
                             })));
}

TEST(QuantizedActivationsOpTest, Tanh) {
  QuantizedActivationsOpModel m(
      BuiltinOperator_TANH,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, -8, 8},
      /*output=*/{TensorType_UINT8, {1, 2, 4, 1}, -1, 1});
  m.SetInput({
      0, -6, 2, 4,   //
      -4, -2, 8, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0, -0.999987, 0.964027, 0.999329,     //
                      -0.996078, -0.96402, 0.99999, 0.76159,  //
                  },
                  4 * (1. / 256))));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({128, 0, 251, 255, 0, 5, 255, 226}));
}

TEST(FloatActivationsOpTest, Sigmoid) {
  FloatActivationsOpModel m(BuiltinOperator_LOGISTIC,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 0.5, 0.002473, 0.880797, 0.982014,       //
                                 0.952574, 0.119203, 0.999955, 0.731059,  //
                             })));
}

TEST(QuantizedActivationsOpTest, Sigmoid) {
  QuantizedActivationsOpModel m(
      BuiltinOperator_LOGISTIC,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, -10, 10});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.5, 0.002473, 0.880797, 0.982014,       //
                      0.952574, 0.119203, 0.999955, 0.731059,  //
                  },
                  kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({128, 1, 227, 251, 244, 32, 255, 188}));
}

TEST(FloatActivationsOpTest, Softmax4D) {
  FloatActivationsOpModel m(0.1,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 1, 4}});
  m.SetInput({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 .23463, .12877, .28658, .35003,  //
                                 .22528, .13664, .45365, .18443,  //
                             })));

  // Same input, but a different shape.
  FloatActivationsOpModel m2(0.1,
                             /*input=*/{TensorType_FLOAT32, {4, 1, 1, 2}});
  m2.SetInput({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                  0.645656, 0.354344,  //
                                  0.450166, 0.549834,  //
                                  0.622459, 0.377541,  //
                                  0.710949, 0.28905,   //
                              })));
}

TEST(QuantizedActivationsOpTest, Softmax4D) {
  QuantizedActivationsOpModel m(
      0.1,
      /*input=*/{TensorType_UINT8, {1, 2, 1, 4}, -10, 10});
  m.SetInput({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(
      0.1,
      /*input=*/{TensorType_UINT8, {4, 1, 1, 2}, -10, 10});
  m2.SetInput({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetDequantizedOutput(), ElementsAreArray(ArrayFloatNear(
                                             {
                                                 0.645656, 0.354344,  //
                                                 0.450166, 0.549834,  //
                                                 0.622459, 0.377541,  //
                                                 0.710949, 0.28905,   //
                                             },
                                             kQuantizedTolerance)));
}

TEST(FloatActivationsOpTest, Softmax2D) {
  FloatActivationsOpModel m(0.1,
                            /*input=*/{TensorType_FLOAT32, {2, 4}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 .23463, .12877, .28658, .35003,  //
                                 .22528, .13664, .45365, .18443,  //
                             })));

  // Same input, but a different shape.
  FloatActivationsOpModel m2(0.1,
                             /*input=*/{TensorType_FLOAT32, {4, 2}});
  m2.SetInput({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                  0.645656, 0.354344,  //
                                  0.450166, 0.549834,  //
                                  0.622459, 0.377541,  //
                                  0.710949, 0.28905,   //
                              })));
}

TEST(QuantizedActivationsOpTest, Softmax2D) {
  QuantizedActivationsOpModel m(0.1,
                                /*input=*/{TensorType_UINT8, {2, 4}, -10, 10});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(0.1,
                                 /*input=*/{TensorType_UINT8, {4, 2}, -10, 10});
  m2.SetInput({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetDequantizedOutput(), ElementsAreArray(ArrayFloatNear(
                                             {
                                                 0.645656, 0.354344,  //
                                                 0.450166, 0.549834,  //
                                                 0.622459, 0.377541,  //
                                                 0.710949, 0.28905,   //
                                             },
                                             kQuantizedTolerance)));
}

// This contains the same test values as the Softmax test, but reference answer
// generated via the following snippet of python:
//   logits1 = tf.constant([[0, -6, 2, 4],[3, -2, 10, 1]], dtype=tf.float32)
//   logits2 = tf.constant([[0,-6],[2,4],[3,-2],[10,1]], dtype=tf.float32)
//   lsm1 = tf.nn.log_softmax(logits1)
//   lsm2 = tf.nn.log_softmax(logits2)
//   with tf.Session() as sess:
//     print('lsm1', sess.run(lsm1))
//     print('lsm2', sess.run(lsm2))

TEST(FloatActivationsOpTest, LogSoftmax) {
  FloatActivationsOpModel m(BuiltinOperator_LOG_SOFTMAX,
                            /*input=*/{TensorType_FLOAT32, {2, 4}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 -4.14297, -10.14297, -2.14297, -.142971,    //
                                 -7.00104, -12.00104, -.00104087, -9.00104,  //
                             })));

  // Same input, but a different shape.
  FloatActivationsOpModel m2(BuiltinOperator_LOG_SOFTMAX,
                             /*input=*/{TensorType_FLOAT32, {4, 2}});
  m2.SetInput({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                  -.00247565, -6.00247,   //
                                  -2.12692, -.126928,     //
                                  -.00671534, -5.00671,   //
                                  -.000123374, -9.00012,  //
                              })));
}

class PReluOpModel : public SingleOpModel {
 public:
  PReluOpModel(const TensorData& input, const TensorData& alpha) {
    input_ = AddInput(input);
    alpha_ = AddInput(alpha);
    output_ = AddOutput(input);
    SetBuiltinOp(BuiltinOperator_PRELU, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_), GetShape(alpha_)});
  }
  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }
  void SetAlpha(std::initializer_list<float> data) {
    PopulateTensor(alpha_, data);
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input_;
  int alpha_;
  int output_;
};

TEST(FloatActivationsOpTest, PRelu) {
  PReluOpModel m({TensorType_FLOAT32, {1, 2, 2, 3}},
                 {TensorType_FLOAT32, {1, 1, 3}});

  m.SetInput({
      0.0f, 0.0f, 0.0f,     // Row 1, Column 1
      1.0f, 1.0f, 1.0f,     // Row 1, Column 2
      -1.0f, -1.0f, -1.0f,  // Row 2, Column 1
      -2.0f, -2.0f, -2.0f,  // Row 1, Column 2
  });
  m.SetAlpha({0.0f, 1.0f, 2.0f});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0.0f, 0.0f, 0.0f,    // Row 1, Column 1
                                 1.0f, 1.0f, 1.0f,    // Row 1, Column 2
                                 0.0f, -1.0f, -2.0f,  // Row 2, Column 1
                                 0.0f, -2.0f, -4.0f,  // Row 1, Column 2
                             }));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
