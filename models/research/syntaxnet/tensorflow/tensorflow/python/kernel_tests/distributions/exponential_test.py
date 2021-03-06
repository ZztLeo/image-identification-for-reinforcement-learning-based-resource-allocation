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
"""Tests for initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import exponential as exponential_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def try_import(name):  # pylint: disable=invalid-name
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


stats = try_import("scipy.stats")


class ExponentialTest(test.TestCase):

  def testExponentialLogPDF(self):
    with session.Session():
      batch_size = 6
      lam = constant_op.constant([2.0] * batch_size)
      lam_v = 2.0
      x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
      exponential = exponential_lib.Exponential(rate=lam)

      log_pdf = exponential.log_prob(x)
      self.assertEqual(log_pdf.get_shape(), (6,))

      pdf = exponential.prob(x)
      self.assertEqual(pdf.get_shape(), (6,))

      if not stats:
        return
      expected_log_pdf = stats.expon.logpdf(x, scale=1 / lam_v)
      self.assertAllClose(log_pdf.eval(), expected_log_pdf)
      self.assertAllClose(pdf.eval(), np.exp(expected_log_pdf))

  def testExponentialCDF(self):
    with session.Session():
      batch_size = 6
      lam = constant_op.constant([2.0] * batch_size)
      lam_v = 2.0
      x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)

      exponential = exponential_lib.Exponential(rate=lam)

      cdf = exponential.cdf(x)
      self.assertEqual(cdf.get_shape(), (6,))

      if not stats:
        return
      expected_cdf = stats.expon.cdf(x, scale=1 / lam_v)
      self.assertAllClose(cdf.eval(), expected_cdf)

  def testExponentialMean(self):
    with session.Session():
      lam_v = np.array([1.0, 4.0, 2.5])
      exponential = exponential_lib.Exponential(rate=lam_v)
      self.assertEqual(exponential.mean().get_shape(), (3,))
      if not stats:
        return
      expected_mean = stats.expon.mean(scale=1 / lam_v)
      self.assertAllClose(exponential.mean().eval(), expected_mean)

  def testExponentialVariance(self):
    with session.Session():
      lam_v = np.array([1.0, 4.0, 2.5])
      exponential = exponential_lib.Exponential(rate=lam_v)
      self.assertEqual(exponential.variance().get_shape(), (3,))
      if not stats:
        return
      expected_variance = stats.expon.var(scale=1 / lam_v)
      self.assertAllClose(exponential.variance().eval(), expected_variance)

  def testExponentialEntropy(self):
    with session.Session():
      lam_v = np.array([1.0, 4.0, 2.5])
      exponential = exponential_lib.Exponential(rate=lam_v)
      self.assertEqual(exponential.entropy().get_shape(), (3,))
      if not stats:
        return
      expected_entropy = stats.expon.entropy(scale=1 / lam_v)
      self.assertAllClose(exponential.entropy().eval(), expected_entropy)

  def testExponentialSample(self):
    with self.test_session():
      lam = constant_op.constant([3.0, 4.0])
      lam_v = [3.0, 4.0]
      n = constant_op.constant(100000)
      exponential = exponential_lib.Exponential(rate=lam)

      samples = exponential.sample(n, seed=137)
      sample_values = samples.eval()
      self.assertEqual(sample_values.shape, (100000, 2))
      self.assertFalse(np.any(sample_values < 0.0))
      if not stats:
        return
      for i in range(2):
        self.assertLess(
            stats.kstest(
                sample_values[:, i], stats.expon(scale=1.0 / lam_v[i]).cdf)[0],
            0.01)

  def testExponentialSampleMultiDimensional(self):
    with self.test_session():
      batch_size = 2
      lam_v = [3.0, 22.0]
      lam = constant_op.constant([lam_v] * batch_size)

      exponential = exponential_lib.Exponential(rate=lam)

      n = 100000
      samples = exponential.sample(n, seed=138)
      self.assertEqual(samples.get_shape(), (n, batch_size, 2))

      sample_values = samples.eval()

      self.assertFalse(np.any(sample_values < 0.0))
      if not stats:
        return
      for i in range(2):
        self.assertLess(
            stats.kstest(
                sample_values[:, 0, i],
                stats.expon(scale=1.0 / lam_v[i]).cdf)[0],
            0.01)
        self.assertLess(
            stats.kstest(
                sample_values[:, 1, i],
                stats.expon(scale=1.0 / lam_v[i]).cdf)[0],
            0.01)

  def testExponentialWithSoftplusRate(self):
    with self.test_session():
      lam = [-2.2, -3.4]
      exponential = exponential_lib.ExponentialWithSoftplusRate(rate=lam)
      self.assertAllClose(nn_ops.softplus(lam).eval(),
                          exponential.rate.eval())


if __name__ == "__main__":
  test.main()
