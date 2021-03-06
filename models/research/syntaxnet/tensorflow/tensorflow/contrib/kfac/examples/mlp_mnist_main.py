# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
r"""Train an MLP on MNIST using K-FAC.

See mlp.py for details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from tensorflow.contrib.kfac.examples import mlp

FLAGS = None


def main(argv):
  _ = argv
  if FLAGS.use_estimator:
    if FLAGS.num_towers != 1:
      raise ValueError("Only 1 device supported in tf.estimator example.")
    mlp.train_mnist_estimator(FLAGS.data_dir, num_epochs=200)
  elif FLAGS.num_towers > 1:
    mlp.train_mnist_multitower(
        FLAGS.data_dir, num_epochs=200, num_towers=FLAGS.num_towers)
  else:
    mlp.train_mnist(FLAGS.data_dir, num_epochs=200)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data_dir",
      type=str,
      default="/tmp/mnist",
      help="Directory to store dataset in.")
  parser.add_argument(
      "--num_towers",
      type=int,
      default=1,
      help="Number of CPUs to split minibatch across.")
  parser.add_argument(
      "--use_estimator",
      action="store_true",
      help="Use tf.estimator API to train.")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
