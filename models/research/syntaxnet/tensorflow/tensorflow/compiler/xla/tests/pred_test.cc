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

// Miscellaneous tests with the PRED type that don't fit anywhere else.
#include <memory>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class PredTest : public ClientLibraryTestBase {
 protected:
  void TestCompare(bool lhs, bool rhs, bool expected,
                   ComputationDataHandle (ComputationBuilder::*op)(
                       const ComputationDataHandle&,
                       const ComputationDataHandle&,
                       tensorflow::gtl::ArraySlice<int64>)) {
    ComputationBuilder builder(client_, TestName());
    ComputationDataHandle lhs_op = builder.ConstantR0<bool>(lhs);
    ComputationDataHandle rhs_op = builder.ConstantR0<bool>(rhs);
    ComputationDataHandle result = (builder.*op)(lhs_op, rhs_op, {});
    ComputeAndCompareR0<bool>(&builder, expected, {});
  }
};

TEST_F(PredTest, ConstantR0PredTrue) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR0<bool>(true);
  ComputeAndCompareR0<bool>(&builder, true, {});
}

TEST_F(PredTest, ConstantR0PredFalse) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR0<bool>(false);
  ComputeAndCompareR0<bool>(&builder, false, {});
}

TEST_F(PredTest, ConstantR0PredCompareEq) {
  TestCompare(true, false, false, &ComputationBuilder::Eq);
}

TEST_F(PredTest, ConstantR0PredCompareNe) {
  TestCompare(true, false, true, &ComputationBuilder::Ne);
}

TEST_F(PredTest, ConstantR0PredCompareLe) {
  TestCompare(true, false, false, &ComputationBuilder::Le);
}

TEST_F(PredTest, ConstantR0PredCompareLt) {
  TestCompare(true, false, false, &ComputationBuilder::Lt);
}

TEST_F(PredTest, ConstantR0PredCompareGe) {
  TestCompare(true, false, true, &ComputationBuilder::Ge);
}

TEST_F(PredTest, ConstantR0PredCompareGt) {
  TestCompare(true, false, true, &ComputationBuilder::Gt);
}

TEST_F(PredTest, ConstantR1Pred) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<bool>({true, false, false, true});
  ComputeAndCompareR1<bool>(&builder, {true, false, false, true}, {});
}

TEST_F(PredTest, ConstantR2Pred) {
  ComputationBuilder builder(client_, TestName());
  auto a =
      builder.ConstantR2<bool>({{false, true, true}, {true, false, false}});
  const string expected = R"(pred[2,3] {
  { 011 },
  { 100 }
})";
  EXPECT_EQ(expected, ExecuteToString(&builder, {}));
}

TEST_F(PredTest, AnyR1True) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<bool>({true, false});
  TF_ASSERT_OK(Any(a, &builder).status());
  ComputeAndCompareR0<bool>(&builder, true, {});
}

TEST_F(PredTest, AnyR1False) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<bool>({false, false});
  TF_ASSERT_OK(Any(a, &builder).status());
  ComputeAndCompareR0<bool>(&builder, false, {});
}

TEST_F(PredTest, AnyR1VacuouslyFalse) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<bool>({});
  TF_ASSERT_OK(Any(a, &builder).status());
  ComputeAndCompareR0<bool>(&builder, false, {});
}

TEST_F(PredTest, AnyR2True) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR2<bool>({
      {false, false, false},
      {false, false, false},
      {false, false, true},
  });
  TF_ASSERT_OK(Any(a, &builder).status());
  ComputeAndCompareR0<bool>(&builder, true, {});
}

TEST_F(PredTest, AnyR2False) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR2<bool>({
      {false, false, false},
      {false, false, false},
      {false, false, false},
  });
  TF_ASSERT_OK(Any(a, &builder).status());
  ComputeAndCompareR0<bool>(&builder, false, {});
}

}  // namespace
}  // namespace xla
