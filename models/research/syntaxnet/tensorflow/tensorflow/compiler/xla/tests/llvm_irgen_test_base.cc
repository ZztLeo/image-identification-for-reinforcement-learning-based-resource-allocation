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

#include "tensorflow/compiler/xla/tests/llvm_irgen_test_base.h"

#include <functional>
#include <utility>

#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {

void LLVMIRGenTestBase::SetIrHook(bool match_optimized_ir) {
  auto llvm_compiler = GetLLVMCompiler();
  using std::placeholders::_1;

  // Add the IR inspection hook to the LLVM compiler.
  if (match_optimized_ir) {
    llvm_compiler->SetPostOptimizationHook(
        std::bind(&LLVMIRGenTestBase::IrHook, this, _1));
  } else {
    llvm_compiler->SetPreOptimizationHook(
        std::bind(&LLVMIRGenTestBase::IrHook, this, _1));
  }
}

void LLVMIRGenTestBase::ResetIrHook() {
  auto llvm_compiler = GetLLVMCompiler();

  llvm_compiler->RemovePreOptimizationHook();
  llvm_compiler->RemovePostOptimizationHook();
}

void LLVMIRGenTestBase::CompileAndVerifyIr(
    std::unique_ptr<HloModule> hlo_module, const string& pattern,
    bool match_optimized_ir) {
  SetIrHook(match_optimized_ir);
  TF_ASSERT_OK(CompileToExecutable(std::move(hlo_module)).status());
  ResetIrHook();

  StatusOr<bool> filecheck_result = RunFileCheck(ir_, pattern);
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(filecheck_result.ValueOrDie());
}

void LLVMIRGenTestBase::CompileAheadOfTimeAndVerifyIr(
    std::unique_ptr<HloModule> hlo_module, const AotCompilationOptions& options,
    const string& pattern, bool match_optimized_ir) {
  SetIrHook(match_optimized_ir);
  ASSERT_TRUE(
      CompileToAotCompilationResult(std::move(hlo_module), options).ok());
  ResetIrHook();

  StatusOr<bool> filecheck_result = RunFileCheck(ir_, pattern);
  ASSERT_TRUE(filecheck_result.ok());
  EXPECT_TRUE(filecheck_result.ValueOrDie());
}

LLVMCompiler* LLVMIRGenTestBase::GetLLVMCompiler() {
  return static_cast<LLVMCompiler*>(backend().compiler());
}

Status LLVMIRGenTestBase::IrHook(const llvm::Module& module) {
  ir_ = llvm_ir::DumpModuleToString(module);
  return Status::OK();
}

}  // namespace xla
