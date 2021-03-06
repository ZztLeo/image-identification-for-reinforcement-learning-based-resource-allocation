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
#include "tensorflow/contrib/lite/testing/tf_driver.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace testing {
namespace {

using ::testing::ElementsAre;

TEST(TfDriverTest, SimpleTest) {
  std::unique_ptr<TfDriver> runner(
      new TfDriver({"a", "b", "c", "d"}, {"float", "float", "float", "float"},
                   {"1,8,8,3", "1,8,8,3", "1,8,8,3", "1,8,8,3"}, {"x", "y"}));

  runner->LoadModel(
      "third_party/tensorflow/contrib/lite/testdata/multi_add.pb");
  EXPECT_TRUE(runner->IsValid()) << runner->GetErrorMessage();

  ASSERT_THAT(runner->GetInputs(), ElementsAre(0, 1, 2, 3));
  ASSERT_THAT(runner->GetOutputs(), ElementsAre(0, 1));

  for (int i : {0, 1, 2, 3}) {
    runner->ReshapeTensor(i, "1,2,2,1");
  }
  ASSERT_TRUE(runner->IsValid());

  runner->SetInput(0, "0.1,0.2,0.3,0.4");
  runner->SetInput(1, "0.001,0.002,0.003,0.004");
  runner->SetInput(2, "0.001,0.002,0.003,0.004");
  runner->SetInput(3, "0.01,0.02,0.03,0.04");
  runner->ResetTensor(2);
  runner->Invoke();

  ASSERT_EQ(runner->ReadOutput(0), "0.101,0.202,0.303,0.404");
  ASSERT_EQ(runner->ReadOutput(1), "0.011,0.022,0.033,0.044");
}

}  // namespace
}  // namespace testing
}  // namespace tflite
