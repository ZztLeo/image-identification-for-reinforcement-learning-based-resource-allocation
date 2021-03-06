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

"""Tests for predictor.core_estimator_predictor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import numpy as np

from tensorflow.contrib.predictor import core_estimator_predictor
from tensorflow.contrib.predictor import testing_common
from tensorflow.python.platform import test


KEYS_AND_OPS = (('sum', lambda x, y: x + y),
                ('product', lambda x, y: x * y,),
                ('difference', lambda x, y: x - y))


class CoreEstimatorPredictorTest(test.TestCase):
  """Test fixture for `CoreEstimatorPredictor`."""

  def setUp(self):
    model_dir = tempfile.mkdtemp()
    self._estimator = testing_common.get_arithmetic_estimator(
        core=True, model_dir=model_dir)
    self._serving_input_receiver_fn = testing_common.get_arithmetic_input_fn(
        core=True, train=False)

  def testDefault(self):
    """Test prediction with default signature."""
    np.random.seed(1111)
    x = np.random.rand()
    y = np.random.rand()
    predictor = core_estimator_predictor.CoreEstimatorPredictor(
        estimator=self._estimator,
        serving_input_receiver_fn=self._serving_input_receiver_fn)
    output = predictor({'x': x, 'y': y})['sum']
    self.assertAlmostEqual(output, x + y, places=3)

  def testSpecifiedSignatureKey(self):
    """Test prediction with spedicified signatures."""
    np.random.seed(1234)
    for output_key, op in KEYS_AND_OPS:
      x = np.random.rand()
      y = np.random.rand()
      expected_output = op(x, y)

      predictor = core_estimator_predictor.CoreEstimatorPredictor(
          estimator=self._estimator,
          serving_input_receiver_fn=self._serving_input_receiver_fn,
          output_key=output_key)
      output_tensor_name = predictor.fetch_tensors[output_key].name
      self.assertRegexpMatches(
          output_tensor_name,
          output_key,
          msg='Unexpected fetch tensor.')
      output = predictor({'x': x, 'y': y})[output_key]
      self.assertAlmostEqual(
          expected_output, output, places=3,
          msg='Failed for output key "{}." '
          'Got output {} for x = {} and y = {}'.format(
              output_key, output, x, y))

if __name__ == '__main__':
  test.main()
