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

"""Tests for LazyAdamOptimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.opt.python.training import lazy_adam_optimizer
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def adam_update_numpy(param,
                      g_t,
                      t,
                      m,
                      v,
                      alpha=0.001,
                      beta1=0.9,
                      beta2=0.999,
                      epsilon=1e-8):
  alpha_t = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * g_t

  param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)
  return param_t, m_t, v_t


class AdamOptimizerTest(test.TestCase):

  def testSparse(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.test_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0_np_indices = np.array([0, 1], dtype=np.int32)
        grads0 = ops.IndexedSlices(
            constant_op.constant(grads0_np),
            constant_op.constant(grads0_np_indices), constant_op.constant([2]))
        grads1_np_indices = np.array([0, 1], dtype=np.int32)
        grads1 = ops.IndexedSlices(
            constant_op.constant(grads1_np),
            constant_op.constant(grads1_np_indices), constant_op.constant([2]))
        opt = lazy_adam_optimizer.LazyAdamOptimizer()
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, beta1_power.eval())
          self.assertAllCloseAccordingToType(0.999**t, beta2_power.eval())
          update.run()

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0.eval())
          self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testSparseDevicePlacement(self):
    for index_dtype in [dtypes.int32, dtypes.int64]:
      with self.test_session(force_gpu=test.is_gpu_available()):
        # If a GPU is available, tests that all optimizer ops can be placed on
        # it (i.e. they have GPU kernels).
        var = variables.Variable([[1.0], [2.0]])
        indices = constant_op.constant([0, 1], dtype=index_dtype)
        gathered_sum = math_ops.reduce_sum(array_ops.gather(var, indices))
        optimizer = lazy_adam_optimizer.LazyAdamOptimizer(3.0)
        minimize_op = optimizer.minimize(gathered_sum)
        variables.global_variables_initializer().run()
        minimize_op.run()

  def testSparseRepeatedIndices(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.test_session():
        repeated_index_update_var = variables.Variable(
            [[1.0], [2.0]], dtype=dtype)
        aggregated_update_var = variables.Variable(
            [[1.0], [2.0]], dtype=dtype)
        grad_repeated_index = ops.IndexedSlices(
            constant_op.constant(
                [0.1, 0.1], shape=[2, 1], dtype=dtype),
            constant_op.constant([1, 1]),
            constant_op.constant([2, 1]))
        grad_aggregated = ops.IndexedSlices(
            constant_op.constant(
                [0.2], shape=[1, 1], dtype=dtype),
            constant_op.constant([1]),
            constant_op.constant([2, 1]))
        repeated_update_opt = lazy_adam_optimizer.LazyAdamOptimizer()
        repeated_update = repeated_update_opt.apply_gradients(
            [(grad_repeated_index, repeated_index_update_var)])
        aggregated_update_opt = lazy_adam_optimizer.LazyAdamOptimizer()
        aggregated_update = aggregated_update_opt.apply_gradients(
            [(grad_aggregated, aggregated_update_var)])
        variables.global_variables_initializer().run()
        self.assertAllClose(aggregated_update_var.eval(),
                            repeated_index_update_var.eval())
        for _ in range(3):
          repeated_update.run()
          aggregated_update.run()
          self.assertAllClose(aggregated_update_var.eval(),
                              repeated_index_update_var.eval())


if __name__ == "__main__":
  test.main()
