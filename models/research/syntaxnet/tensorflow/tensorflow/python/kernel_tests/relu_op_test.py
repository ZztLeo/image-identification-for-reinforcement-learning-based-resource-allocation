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
"""Tests for Relu and ReluGrad."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


def _elu_grad_grad(activation):
  if activation < 0:
    return np.exp(activation)
  return 0


class ReluTest(test.TestCase):

  def _npRelu(self, np_features):
    return np.maximum(np_features, np.zeros(np_features.shape))

  def testNpRelu(self):
    self.assertAllClose(
        np.array([[0.0, 0.7, 0.0, 0.3, 0.0], [0.1, 0.0, 0.5, 0.0, 0.9]]),
        self._npRelu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                     0.9]])))

  def _testRelu(self, np_features, use_gpu=False):
    np_relu = self._npRelu(np_features)
    with self.test_session(use_gpu=use_gpu):
      relu = nn_ops.relu(np_features)
      tf_relu = relu.eval()
    self.assertAllClose(np_relu, tf_relu)
    self.assertShapeEqual(np_relu, relu)

  def testNumbers(self):
    for t in [np.int32, np.int64, np.float16, np.float32, np.float64]:
      self._testRelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=False)
      if t in [np.float16, np.float32, np.float64]:
        self._testRelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
            use_gpu=True)

  # The gradient test for ReLU is a bit tricky as the derivative is not well
  # defined at around zero and we want to avoid that in terms of input values.
  def testGradientFloat32(self):
    with self.test_session():
      x = constant_op.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5],
          name="x")
      y = nn_ops.relu(x, name="relu")
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], y, [2, 5], x_init_value=x_init)
    print("relu (float32) gradient err = ", err)
    self.assertLess(err, 1e-4)

  # The gradient for fp16 is inaccurate due to the low-precision.
  # Instead of relying on compute_gradient_error, we compare the fp16 analytical
  # gradient against their fp32 counterpart.
  def testGradientFloat16(self):
    with self.test_session(use_gpu=True) as sess:
      # Randomly construct a 1D shape from [1, 40)
      shape = random_ops.random_uniform(
          [1], minval=1, maxval=40, dtype=dtypes.int32)

      # Construct the fp32 graph and its gradient.
      x = random_ops.random_uniform(shape, minval=-1, maxval=1, name="x")
      y1 = nn_ops.relu(x, name="relu_fp32")
      l1 = nn_ops.l2_loss(y1)
      dx_f32 = gradients_impl.gradients(l1, x)

      # Construct the fp16 graph and its gradient.
      # It starts with the same x, in fp32. But before it reaches Relu, it is
      # cast into fp16. So during backprop, the gradient computation is in fp16.
      x2 = math_ops.cast(x, dtype=dtypes.float16, name="cast")
      y2 = nn_ops.relu(x2, name="relu_fp16")
      l2 = nn_ops.l2_loss(y2)
      dx_f16 = gradients_impl.gradients(l2, x)

      # Repeat the experiment for 100 times. All tensor shapes and its tensor
      # values are randomly generated for each run.
      for _ in xrange(100):
        dx_f32_v, dx_f16_v = sess.run([dx_f32, dx_f16])
        self.assertAllClose(dx_f32_v, dx_f16_v, atol=3e-4)

  def testGradientFloat64(self):
    with self.test_session():
      x = constant_op.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5],
          dtype=dtypes.float64,
          name="x")
      y = nn_ops.relu(x, name="relu")
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], y, [2, 5], x_init_value=x_init)
    print("relu (float64) gradient err = ", err)
    self.assertLess(err, 1e-10)

  def testGradGradFloat32(self):
    with self.test_session():
      x = constant_op.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5],
          name="x")
      y = nn_ops.relu(x, name="relu")
      z = gradients_impl.gradients(y, x)
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], z[0], [2, 5], x_init_value=x_init)
    print("relu (float32) gradient of gradient err = ", err)
    self.assertLess(err, 1e-4)

  def testGradGradFloat64(self):
    with self.test_session():
      x = constant_op.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5],
          dtype=dtypes.float64,
          name="x")
      y = nn_ops.relu(x, name="relu")
      z = gradients_impl.gradients(y, x)
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], z[0], [2, 5], x_init_value=x_init)
    print("relu (float64) gradient of gradient err = ", err)
    self.assertLess(err, 1e-10)

  def testGradientScalar(self):
    with self.test_session() as sess:
      x = variables.Variable(100.)
      y = nn_ops.relu(x)
      loss = y**2
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.25)
      train_op = optimizer.minimize(loss)
      sess.run(variables.global_variables_initializer())
      sess.run(train_op)
      self.assertAllClose(x.eval(), 50.0)


class Relu6Test(test.TestCase):

  def _npRelu6(self, np_features):
    sixes = np.copy(np_features)
    sixes.fill(6.0)
    return np.minimum(
        np.maximum(np_features, np.zeros(np_features.shape)), sixes)

  def testNpRelu6(self):
    self.assertAllClose(
        np.array([[0.0, 0.7, 0.0, 0.3, 6.0], [0.1, 0.0, 6.0, 0.0, 0.9]]),
        self._npRelu6(
            np.array([[-0.9, 0.7, -0.5, 0.3, 6.0], [0.1, -0.3, 6.5, -0.7,
                                                    0.9]])))

  def _testRelu6(self, np_features, use_gpu=False):
    np_relu6 = self._npRelu6(np_features)
    with self.test_session(use_gpu=use_gpu):
      relu6 = nn_ops.relu6(np_features)
      tf_relu6 = relu6.eval()
    self.assertAllClose(np_relu6, tf_relu6)
    self.assertShapeEqual(np_relu6, relu6)

  def testNumbers(self):
    for t in [np.int32, np.int64, np.float16, np.float32, np.float64]:
      self._testRelu6(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=False)
      if t in [np.float16, np.float, np.double]:
        self._testRelu6(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
            use_gpu=True)

  # The gradient test for ReLU6 is a bit tricky as the derivative is
  # not well defined at around zero and six and we want to avoid that
  # in terms of input values.
  def testGradientFloat32(self):
    with self.test_session():
      x = constant_op.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 6.1, 6.3, 6.5, 6.7, 6.9],
          shape=[2, 5],
          name="x")
      y = nn_ops.relu6(x, name="relu6")
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], y, [2, 5], x_init_value=x_init)
    print("relu6 (float32) gradient err = ", err)
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    with self.test_session():
      x = constant_op.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 6.1, 6.3, 6.5, 6.7, 6.9],
          shape=[2, 5],
          dtype=dtypes.float64,
          name="x")
      y = nn_ops.relu6(x, name="relu6")
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], y, [2, 5], x_init_value=x_init)
    print("relu6 (float64) gradient err = ", err)
    self.assertLess(err, 1e-10)


class EluTest(test.TestCase):

  def _npElu(self, np_features):
    return np.where(np_features < 0, np.exp(np_features) - 1, np_features)

  def testNpElu(self):
    self.assertAllClose(
        np.array([[-0.59343034025, 0.7, -0.39346934028, 0.3, -0.09516258196],
                  [0.1, -0.25918177931, 0.5, -0.5034146962, 0.9]]),
        self._npElu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                     0.9]])))

  def _testElu(self, np_features, use_gpu=False):
    np_elu = self._npElu(np_features)
    with self.test_session(use_gpu=use_gpu):
      elu = nn_ops.elu(np_features)
      tf_elu = elu.eval()
    self.assertAllClose(np_elu, tf_elu)
    self.assertShapeEqual(np_elu, elu)

  def testNumbers(self):
    for t in [np.float16, np.float32, np.float64]:
      self._testElu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=False)
      self._testElu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=True)

  def testGradientFloat32(self):
    with self.test_session():
      x_val = [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
      x = constant_op.constant(x_val, name="x")
      y = nn_ops.elu(x, name="elu")
      x_init = np.asarray(x_val, dtype=np.float32, order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], y, [2, 5], x_init_value=x_init)
    print("elu (float32) gradient err = ", err)
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    with self.test_session():
      x_val = [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
      x = constant_op.constant(x_val, dtype=dtypes.float64, name="x")
      y = nn_ops.elu(x, name="elu")
      x_init = np.asarray(x_val, dtype=np.float64, order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], y, [2, 5], x_init_value=x_init)
    print("elu (float64) gradient err = ", err)
    self.assertLess(err, 1e-6)

  def testGradGrad(self):
    with self.test_session():
      x = array_ops.placeholder(dtype=dtypes.float32)
      elu = nn_ops.elu(x)
      g, = gradients_impl.gradients(elu, x)
      gg, = gradients_impl.gradients(g, x)

      for x_val in [-1, -0.5, 0.5, 1]:
        err = np.abs(gg.eval(feed_dict={x: x_val}) - _elu_grad_grad(x_val))
        self.assertLess(err, 1e-4)

  def testGradGradFloat32(self):
    with self.test_session():
      x = constant_op.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5],
          name="x")
      y = nn_ops.elu(x, name="elu")
      z = gradients_impl.gradients(y, x)
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], z[0], [2, 5], x_init_value=x_init)
    print("elu (float32) gradient of gradient err = ", err)
    self.assertLess(err, 1e-4)

  def testGradGradFloat64(self):
    with self.test_session():
      x = constant_op.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5],
          dtype=dtypes.float64,
          name="x")
      y = nn_ops.elu(x, name="elu")
      z = gradients_impl.gradients(y, x)
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], z[0], [2, 5], x_init_value=x_init)
    print("elu (float64) gradient of gradient err = ", err)
    self.assertLess(err, 1e-6)


class SeluTest(test.TestCase):

  def _npSelu(self, np_features):
    scale = 1.0507009873554804934193349852946
    scale_alpha = 1.7580993408473768599402175208123
    return np.where(np_features < 0, scale_alpha * (np.exp(np_features) - 1),
                    scale * np_features)

  def testNpSelu(self):
    self.assertAllClose(
        np.array([[-1.0433095, 0.73549069, -0.6917582, 0.3152103, -0.16730527],
                  [0.1050701, -0.45566732, 0.5253505, -0.88505305, 0.9456309]]),
        self._npSelu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                     0.9]])))

  def _testSelu(self, np_features, use_gpu=False):
    np_selu = self._npSelu(np_features)
    with self.test_session(use_gpu=use_gpu):
      selu = nn_ops.selu(np_features)
      tf_selu = selu.eval()
    self.assertAllClose(np_selu, tf_selu)
    self.assertShapeEqual(np_selu, selu)

  def testNumbers(self):
    for t in [np.float16, np.float32, np.float64]:
      self._testSelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=False)
      self._testSelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=True)

  def testGradientFloat32(self):
    with self.test_session():
      x_val = [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
      x = constant_op.constant(x_val, name="x")
      y = nn_ops.selu(x, name="selu")
      x_init = np.asarray(x_val, dtype=np.float32, order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], y, [2, 5], x_init_value=x_init)
    print("selu (float32) gradient err = ", err)
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    with self.test_session():
      x_val = [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
      x = constant_op.constant(x_val, dtype=dtypes.float64, name="x")
      y = nn_ops.selu(x, name="selu")
      x_init = np.asarray(x_val, dtype=np.float64, order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], y, [2, 5], x_init_value=x_init)
    print("selu (float64) gradient err = ", err)
    self.assertLess(err, 1e-6)

  def testGradGradFloat32(self):
    with self.test_session():
      x = constant_op.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5],
          name="x")
      y = nn_ops.selu(x, name="selu")
      z = gradients_impl.gradients(y, x)
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], z[0], [2, 5], x_init_value=x_init)
    print("selu (float32) gradient of gradient err = ", err)
    self.assertLess(err, 1e-4)

  def testGradGradFloat64(self):
    with self.test_session():
      x = constant_op.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5],
          dtype=dtypes.float64,
          name="x")
      y = nn_ops.selu(x, name="selu")
      z = gradients_impl.gradients(y, x)
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], z[0], [2, 5], x_init_value=x_init)
    print("selu (float64) gradient of gradient err = ", err)
    self.assertLess(err, 1e-6)


class CreluTest(test.TestCase):

  def testCreluShape(self):
    f = random_ops.random_normal([50, 5, 7, 10])
    t = nn_ops.crelu(f)
    self.assertEqual([50, 5, 7, 20], t.get_shape())

  def _testCrelu(self, np_features, use_gpu=False):
    np_relu = np.maximum(np_features, np.zeros_like(np_features))
    np_neg_relu = np.maximum(-np_features, np.zeros_like(np_features))
    np_crelu = np.concatenate((np_relu, np_neg_relu),
                              len(np_features.shape) - 1)

    with self.test_session(use_gpu=use_gpu):
      crelu = nn_ops.crelu(np_features)
      tf_relu = crelu.eval()

    self.assertAllClose(np_crelu, tf_relu)
    self.assertShapeEqual(np_crelu, crelu)

  def testNumbers(self):
    for t in [np.int32, np.int64, np.float16, np.float32, np.float64]:
      self._testCrelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=False)
      if t in [np.float16, np.float32, np.float64]:
        self._testCrelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
            use_gpu=True)

  def testNumbersWithAxis0(self):
    with self.test_session():
      crelu = nn_ops.crelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]), axis=0)
      tf_relu = crelu.eval()
      np_crelu = np.array([[0, 7, 0, 3, 0], [1, 0, 5, 0, 9], [9, 0, 5, 0, 1],
                           [0, 3, 0, 7, 0]])
      self.assertAllEqual(np_crelu, tf_relu)

  def testNumbersWithAxis1(self):
    with self.test_session():
      crelu = nn_ops.crelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]), axis=1)
      tf_relu = crelu.eval()
      np_crelu = np.array([[0, 7, 0, 3, 0, 9, 0, 5, 0, 1],
                           [1, 0, 5, 0, 9, 0, 3, 0, 7, 0]])
      self.assertAllEqual(np_crelu, tf_relu)


if __name__ == "__main__":
  test.main()
