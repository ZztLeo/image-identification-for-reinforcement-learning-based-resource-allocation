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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MATCHERS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MATCHERS_H_

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/test.h"

namespace xla {
namespace testing {

class HloMatcher : public ::testing::MatcherInterface<const HloInstruction*> {
 public:
  HloMatcher(HloOpcode opcode,
             std::vector<::testing::Matcher<const HloInstruction*>> operands)
      : opcode_(opcode), operands_(operands) {}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;

  void DescribeTo(::std::ostream* os) const override;

 private:
  HloOpcode opcode_;
  std::vector<::testing::Matcher<const HloInstruction*>> operands_;
};

// Custom matcher for parameters, which accepts a parameter number.
class HloParameterMatcher : public HloMatcher {
 public:
  explicit HloParameterMatcher(int64 parameter_number)
      : HloMatcher(HloOpcode::kParameter, /*operands=*/{}),
        parameter_number_(parameter_number) {}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;

 private:
  int64 parameter_number_;
};

// Custom matcher for get-tuple-element instructions, which accepts a tuple
// index to match.
class HloGetTupleElementMatcher : public HloMatcher {
 public:
  HloGetTupleElementMatcher(::testing::Matcher<const HloInstruction*> operand,
                            int64 tuple_index)
      : HloMatcher(HloOpcode::kGetTupleElement, /*operands=*/{operand}),
        tuple_index_(tuple_index) {}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;

 private:
  int64 tuple_index_;
};

// Custom matcher for custom-call instructions, which accepts a matcher for its
// call target.
class HloCustomCallMatcher : public HloMatcher {
 public:
  HloCustomCallMatcher(
      ::testing::Matcher<string> call_target_matcher,
      std::vector<::testing::Matcher<const HloInstruction*>> operands)
      : HloMatcher(HloOpcode::kCustomCall, operands),
        call_target_matcher_(call_target_matcher) {}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  ::testing::Matcher<string> call_target_matcher_;
};

// HloInstruction* matchers for opcode and operands. Example:
//   namespace op = xla::opcode_matchers;
//   EXPECT_THAT(instruction,
//               op::Add(op::Reshape(), op::Add(op::Reshape(), _)));
namespace opcode_matchers {
#define HLO_MATCHER(opcode)                                                \
  template <typename... M>                                                 \
  ::testing::Matcher<const ::xla::HloInstruction*> opcode(M... operands) { \
    return ::testing::MakeMatcher(new ::xla::testing::HloMatcher(          \
        ::xla::HloOpcode::k##opcode, {operands...}));                      \
  }
HLO_MATCHER(Abs);
HLO_MATCHER(Add);
HLO_MATCHER(Bitcast);
HLO_MATCHER(Broadcast);
HLO_MATCHER(BatchNormGrad);
HLO_MATCHER(Call);
HLO_MATCHER(Ceil);
HLO_MATCHER(Clamp);
HLO_MATCHER(Concatenate);
HLO_MATCHER(Conditional);
HLO_MATCHER(Constant);
HLO_MATCHER(Convert);
HLO_MATCHER(Convolution);
HLO_MATCHER(Copy);
HLO_MATCHER(CrossReplicaSum);
HLO_MATCHER(Divide);
HLO_MATCHER(Dot);
HLO_MATCHER(DynamicSlice);
HLO_MATCHER(DynamicUpdateSlice);
HLO_MATCHER(Eq);
HLO_MATCHER(Exp);
HLO_MATCHER(Floor);
HLO_MATCHER(Fusion);
HLO_MATCHER(Ge);
HLO_MATCHER(Gt);
HLO_MATCHER(Infeed);
HLO_MATCHER(IsFinite);
HLO_MATCHER(Le);
HLO_MATCHER(Log);
HLO_MATCHER(And);
HLO_MATCHER(Not);
HLO_MATCHER(Or);
HLO_MATCHER(Lt);
HLO_MATCHER(Map);
HLO_MATCHER(Maximum);
HLO_MATCHER(Minimum);
HLO_MATCHER(Multiply);
HLO_MATCHER(Ne);
HLO_MATCHER(Negate);
HLO_MATCHER(Outfeed);
HLO_MATCHER(Pad);
HLO_MATCHER(Power);
HLO_MATCHER(Recv);
HLO_MATCHER(RecvDone);
HLO_MATCHER(Reduce);
HLO_MATCHER(ReducePrecision);
HLO_MATCHER(ReduceWindow);
HLO_MATCHER(Remainder);
HLO_MATCHER(Reshape);
HLO_MATCHER(Reverse);
HLO_MATCHER(Rng);
HLO_MATCHER(Select);
HLO_MATCHER(SelectAndScatter);
HLO_MATCHER(Send);
HLO_MATCHER(SendDone);
HLO_MATCHER(ShiftLeft);
HLO_MATCHER(ShiftRightLogical);
HLO_MATCHER(ShiftRightArithmetic);
HLO_MATCHER(Sign);
HLO_MATCHER(Slice);
HLO_MATCHER(Sort);
HLO_MATCHER(Subtract);
HLO_MATCHER(Tanh);
HLO_MATCHER(Trace);
HLO_MATCHER(Transpose);
HLO_MATCHER(Tuple);
HLO_MATCHER(While);

// The special cases below let you check additional information about the
// HloInstruction, beyond just its opcode and operands.  In all cases you can
// still use the generic matcher which doesn't check this info.
//
// Feel free to add additional custom matchers below.

//  - Parameter(N) matches parameter number N.
//  - Parameter() matches any parameter.
inline ::testing::Matcher<const ::xla::HloInstruction*> Parameter(
    int64 parameter_number) {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloParameterMatcher(parameter_number));
}
inline ::testing::Matcher<const ::xla::HloInstruction*> Parameter() {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloMatcher(HloOpcode::kParameter, {}));
}

// GetTupleElement(operand, N) matches a GTE instruction which gets the N'th
// tuple element of operand, while GetTupleElement(operand) matches any GTE
// operation on operand, and GetTupleElement() matches any GTE operation at all.
inline ::testing::Matcher<const ::xla::HloInstruction*> GetTupleElement(
    ::testing::Matcher<const HloInstruction*> operand, int64 tuple_index) {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloGetTupleElementMatcher(operand, tuple_index));
}
inline ::testing::Matcher<const ::xla::HloInstruction*> GetTupleElement(
    ::testing::Matcher<const HloInstruction*> operand) {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloMatcher(HloOpcode::kGetTupleElement, {operand}));
}
inline ::testing::Matcher<const ::xla::HloInstruction*> GetTupleElement() {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloMatcher(HloOpcode::kGetTupleElement, {}));
}

// - CustomCall(T, operand1, ..., operandN) matches a CustomCall with call
//   target T and the given operands.
//
// - CustomCall(operand1, ..., operandN) matches any CustomCall HLO with the
//   given operands.
//
// - CustomCall() matches any CustomCall HLO at all.
template <typename... M>
inline ::testing::Matcher<const ::xla::HloInstruction*> CustomCall(
    ::testing::Matcher<string> call_target_matcher, M... operands) {
  return ::testing::MakeMatcher(new ::xla::testing::HloCustomCallMatcher(
      call_target_matcher, {operands...}));
}
// This overload of CustomCall(A, B, C, ...) exists iff A is not convertible to
// ::testing::Matcher<string>.  In that case, we want to prefer the overload
// above.
template <typename FirstM, typename... M,
          typename Dummy = typename std::enable_if<
              !std::is_convertible<FirstM, ::testing::Matcher<string>>::value,
              void>::type*>
inline ::testing::Matcher<const ::xla::HloInstruction*> CustomCall(
    FirstM operands_first, M... operands_rest) {
  return ::testing::MakeMatcher(new ::xla::testing::HloMatcher(
      HloOpcode::kCustomCall, {operands_first, operands_rest...}));
}
inline ::testing::Matcher<const ::xla::HloInstruction*> CustomCall() {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloMatcher(HloOpcode::kCustomCall, {}));
}

#undef HLO_MATCHER
}  // namespace opcode_matchers

// Helper to convert smart to raw pointers for matching.
template <typename Container>
std::vector<const HloInstruction*> Pointers(const Container& container) {
  std::vector<const HloInstruction*> result;
  result.reserve(container.size());
  for (const auto& entry : container) result.push_back(entry.get());
  return result;
}

}  // namespace testing

// Tell GMock to print HloInstruction* by value, so error messages are nice.
// Has to be in the same namespace as 'HloInstruction'.
void PrintTo(const HloInstruction* inst, ::std::ostream* os);
void PrintTo(HloInstruction* inst, ::std::ostream* os);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MATCHERS_H_
