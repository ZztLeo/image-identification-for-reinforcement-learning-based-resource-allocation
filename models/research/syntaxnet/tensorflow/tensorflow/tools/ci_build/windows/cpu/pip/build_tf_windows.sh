#!/bin/bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script assumes the standard setup on tensorflow Jenkins windows machines.
# It is NOT guaranteed to work on any other machine. Use at your own risk!
#
# REQUIREMENTS:
# * All installed in standard locations:
#   - JDK8, and JAVA_HOME set.
#   - Microsoft Visual Studio 2015 Community Edition
#   - Msys2
#   - Anaconda3
# * Bazel windows executable copied as "bazel.exe" and included in PATH.

# All commands shall pass, and all should be visible.
set -x
set -e

# This script is under <repo_root>/tensorflow/tools/ci_build/windows/cpu/pip/
# Change into repository root.
script_dir=$(dirname $0)
cd ${script_dir%%tensorflow/tools/ci_build/windows/cpu/pip}.

# Setting up the environment variables Bazel and ./configure needs
source "tensorflow/tools/ci_build/windows/bazel/common_env.sh" \
  || { echo "Failed to source common_env.sh" >&2; exit 1; }

# load bazel_test_lib.sh
source "tensorflow/tools/ci_build/windows/bazel/bazel_test_lib.sh" \
  || { echo "Failed to source bazel_test_lib.sh" >&2; exit 1; }

skip_test=0

for ARG in "$@"; do
  if [[ "$ARG" == --skip_test ]]; then
    skip_test=1
  fi
done

run_configure_for_cpu_build

# --define=override_eigen_strong_inline=true speeds up the compiling of conv_grad_ops_3d.cc and conv_ops_3d.cc
# by 20 minutes. See https://github.com/tensorflow/tensorflow/issues/10521
BUILD_OPTS="--define=override_eigen_strong_inline=true"
bazel build -c opt $BUILD_OPTS tensorflow/tools/pip_package:build_pip_package || exit $?

if [[ "$skip_test" == 1 ]]; then
  exit 0
fi

# Create a python test directory to avoid package name conflict
PY_TEST_DIR="py_test_dir"
create_python_test_dir "${PY_TEST_DIR}"

./bazel-bin/tensorflow/tools/pip_package/build_pip_package "$PWD/${PY_TEST_DIR}"

# Running python tests on Windows needs pip package installed
PIP_NAME=$(ls ${PY_TEST_DIR}/tensorflow-*.whl)
reinstall_tensorflow_pip ${PIP_NAME}

# Define no_tensorflow_py_deps=true so that every py_test has no deps anymore,
# which will result testing system installed tensorflow
bazel test -c opt $BUILD_OPTS -k --test_output=errors \
  --define=no_tensorflow_py_deps=true --test_lang_filters=py \
  --test_tag_filters=-no_pip,-no_windows,-no_oss \
  --build_tag_filters=-no_pip,-no_windows,-no_oss --build_tests_only \
  --flaky_test_attempts=3 \
  //${PY_TEST_DIR}/tensorflow/python/... \
  //${PY_TEST_DIR}/tensorflow/contrib/...
