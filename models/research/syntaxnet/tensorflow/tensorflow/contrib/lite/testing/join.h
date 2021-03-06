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
#ifndef TENSORFLOW_CONTRIB_LITE_TESTING_JOIN_H_
#define TENSORFLOW_CONTRIB_LITE_TESTING_JOIN_H_

#include <cstdlib>
#include <sstream>
#include <string>

namespace tflite {
namespace testing {

// Join a list of data separated by delimieter.
template <typename T>
string Join(T* data, size_t len, const string& delimiter) {
  if (len == 0 || data == nullptr) {
    return "";
  }
  std::stringstream result;
  result << data[0];
  for (int i = 1; i < len; i++) {
    result << delimiter << data[i];
  }
  return result.str();
}

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_TESTING_JOIN_H_
