#!/usr/bin/env bash
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
# Script to generate tarballs:
# (1) The TensorFlow C-library: Containing C API header files and libtensorflow.so
# (2) Native library for the TensorFlow Java API: Containing libtensorflow_jni.so
# And jars:
# (3) Java API .jar
# (4) Java API sources .jar
#
# These binary distributions will allow use of TensorFlow in various languages
# without having to compile the TensorFlow framework from sources, which takes
# a while and also introduces many other dependencies.
#
# Usage:
# - Source this file in another bash script
# - Execute build_libtensorflow_tarball SUFFIX
#
# Produces:
# - lib_package/libtensorflow${SUFFIX}.tar.gz
# - lib_package/libtensorflow_jni${SUFFIX}.tar.gz
# - lib_package/libtensorflow.jar
# - lib_package/libtensorflow-src.jar
# - lib_package/libtensorflow_proto.zip
#
# ASSUMPTIONS:
# - build_libtensorflow_tarball is invoked from the root of the git tree.
# - Any environment variables needed by the "configure" script have been set.

function build_libtensorflow_tarball() {
  # Sanity check that this is being run from the root of the git repository.
  if [ ! -e "WORKSPACE" ]; then
    echo "Must run this from the root of the bazel workspace"
    exit 1
  fi
  # Delete any leftovers from previous builds in this workspace.
  DIR=lib_package
  rm -rf ${DIR}

  TARBALL_SUFFIX="${1}"
  BAZEL_OPTS="-c opt --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0"
  export CC_OPT_FLAGS='-mavx'
  if [ "${TF_NEED_CUDA}" == "1" ]; then
    BAZEL_OPTS="${BAZEL_OPTS} --config=cuda"
  fi
  bazel clean --expunge
  yes "" | ./configure

  # Remove this test call when
  # https://github.com/bazelbuild/bazel/issues/2352
  # and https://github.com/bazelbuild/bazel/issues/1580
  # have been resolved and the "manual" tags on the BUILD targets
  # in tensorflow/tools/lib_package/BUILD are removed.
  # Till then, must manually run the test since these tests are
  # not covered by the continuous integration.
  bazel test ${BAZEL_OPTS} \
    //tensorflow/tools/lib_package:libtensorflow_test \
    //tensorflow/tools/lib_package:libtensorflow_java_test

  bazel build ${BAZEL_OPTS} \
    //tensorflow/tools/lib_package:libtensorflow.tar.gz \
    //tensorflow/tools/lib_package:libtensorflow_jni.tar.gz \
    //tensorflow/java:libtensorflow.jar \
    //tensorflow/java:libtensorflow-src.jar \
    //tensorflow/tools/lib_package:libtensorflow_proto.zip

  mkdir -p ${DIR}

  cp bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz ${DIR}/libtensorflow${TARBALL_SUFFIX}.tar.gz
  cp bazel-bin/tensorflow/tools/lib_package/libtensorflow_jni.tar.gz ${DIR}/libtensorflow_jni${TARBALL_SUFFIX}.tar.gz
  cp bazel-bin/tensorflow/java/libtensorflow.jar ${DIR}
  cp_normalized_srcjar bazel-bin/tensorflow/java/libtensorflow-src.jar ${DIR}/libtensorflow-src.jar
  cp bazel-genfiles/tensorflow/tools/lib_package/libtensorflow_proto.zip ${DIR}
  chmod -x ${DIR}/*
}

# Helper function to copy a srcjar after moving any source files
# directly under the root to the "maven-style" src/main/java layout
#
# Source files generated by annotation processors appear directly
# under the root of srcjars jars created by bazel, rather than under
# the maven-style src/main/java subdirectory.
#
# Bazel manages annotation generated source as follows: First, it
# calls javac with options that create generated files under a
# bazel-out directory. Next, it archives the generated source files
# into a srcjar directly under the root. There doesn't appear to be a
# simple way to parameterize this from bazel, hence this helper to
# "normalize" the srcjar layout.
#
# Arguments:
#   src_jar - path to the original srcjar
#   dest_jar - path to the destination
# Returns:
#   None
function cp_normalized_srcjar() {
  local src_jar="$1"
  local dest_jar="$2"
  if [[ -z "${src_jar}" || -z "${dest_jar}" ]]; then
    echo "Unexpected: missing arguments" >&2
    exit 2
  fi
  local tmp_dir
  tmp_dir=$(mktemp -d)
  cp "${src_jar}" "${tmp_dir}/orig.jar"
  pushd "${tmp_dir}"
  # Extract any src/ files
  jar -xf "${tmp_dir}/orig.jar" src/
  # Extract any org/ files under src/main/java
  (mkdir -p src/main/java && cd src/main/java && jar -xf "${tmp_dir}/orig.jar" org/)
  # Repackage src/
  jar -cMf "${tmp_dir}/new.jar" src
  popd
  cp "${tmp_dir}/new.jar" "${dest_jar}"
  rm -rf "${tmp_dir}"
}
