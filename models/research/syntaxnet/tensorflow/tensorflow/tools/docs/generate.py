# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Generate docs for the TensorFlow Python API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

from tensorflow.python import debug as tf_debug
from tensorflow.python.util import tf_inspect
from tensorflow.tools.docs import generate_lib

if __name__ == '__main__':
  doc_generator = generate_lib.DocGenerator()
  doc_generator.add_output_dir_argument()
  doc_generator.add_src_dir_argument()

  # This doc generator works on the TensorFlow codebase. Since this script lives
  # at tensorflow/tools/docs, and all code is defined somewhere inside
  # tensorflow/, we can compute the base directory (two levels up), which is
  # valid unless we're trying to apply this to a different code base, or are
  # moving the script around.
  script_dir = os.path.dirname(tf_inspect.getfile(tf_inspect.currentframe()))
  default_base_dir = os.path.join(script_dir, '..', '..')
  doc_generator.add_base_dir_argument(default_base_dir)

  flags = doc_generator.parse_known_args()

  # Suppress documentation of some symbols that users should never use.
  del tf.layers.Layer.inbound_nodes
  del tf.layers.Layer.outbound_nodes

  # tf_debug is not imported with tf, it's a separate module altogether
  doc_generator.set_py_modules([('tf', tf), ('tfdbg', tf_debug)])

  sys.exit(doc_generator.build(flags))
