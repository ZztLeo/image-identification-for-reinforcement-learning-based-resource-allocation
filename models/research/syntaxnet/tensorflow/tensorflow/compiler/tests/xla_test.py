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
"""Definition of XLA test case."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import random
import re

import numpy as np

from tensorflow.contrib.compiler import jit
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import flags
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging

FLAGS = flags.FLAGS

flags.DEFINE_string('test_device', None,
                    'Tensorflow device on which to place operators under test')
flags.DEFINE_string('types', None, 'Types to test. Comma-separated list.')
flags.DEFINE_string('disabled_manifest', None,
                    'Path to a file with a list of tests that should not run.')
flags.DEFINE_string('tf_xla_flags', None,
                    'Value to set the TF_XLA_FLAGS environment variable to')


class XLATestCase(test.TestCase):
  """XLA test cases are parameterized test cases."""

  def __init__(self, method_name='runTest'):
    super(XLATestCase, self).__init__(method_name)
    self.device = FLAGS.test_device
    self.has_custom_call = (self.device == 'XLA_CPU')
    self._all_tf_types = set([
        dtypes.as_dtype(types_pb2.DataType.Value(name))
        for name in FLAGS.types.split(',')
    ])
    self.int_tf_types = set([
        dtype for dtype in self._all_tf_types if dtype.is_integer
    ])
    self._float_tf_types = set([
        dtype for dtype in self._all_tf_types if dtype.is_floating
    ])
    self.complex_tf_types = set([
        dtype for dtype in self._all_tf_types if dtype.is_complex
    ])
    self._numeric_tf_types = set(
        self.int_tf_types | self._float_tf_types | self.complex_tf_types)

    self._all_types = set(
        [dtype.as_numpy_dtype for dtype in self._all_tf_types])
    self._int_types = set([dtype.as_numpy_dtype for dtype in self.int_tf_types])
    self._float_types = set(
        [dtype.as_numpy_dtype for dtype in self._float_tf_types])
    self.complex_types = set([
        dtype.as_numpy_dtype for dtype in self.complex_tf_types
    ])
    self._numeric_types = set(self._int_types | self._float_types
                              | self.complex_types)

    # Parse the manifest file, if any, into a regex identifying tests to
    # disable
    self.disabled_regex = None
    self._method_types_filter = dict()
    # TODO(xpan): Make it text proto if it doesn't scale.
    # Each line of the manifest file specifies an entry. The entry can be
    # 1) TestNameRegex  // E.g. CumprodTest.* Or
    # 2) TestName TypeName  // E.g. AdamOptimizerTest.testSharing DT_BFLOAT16
    # The 1) disables the entire test. While 2) only filter some numeric types
    # so that they are not used in those tests.

    if FLAGS.disabled_manifest is not None:
      comments_re = re.compile('#.*$')
      manifest_file = open(FLAGS.disabled_manifest, 'r')
      disabled_tests = []
      disabled_method_types = []
      for l in manifest_file.read().splitlines():
        if not l:
          continue
        entry = comments_re.sub('', l).strip().split(' ')
        if len(entry) == 1:
          disabled_tests.append(entry[0])
        elif len(entry) == 2:
          disabled_method_types.append(
              (entry[0], entry[1].strip().split(',')))
        else:
          raise ValueError('Bad entry in manifest file.')

      self.disabled_regex = re.compile('|'.join(disabled_tests))
      for method, types in disabled_method_types:
        self._method_types_filter[method] = set([
            dtypes.as_dtype(types_pb2.DataType.Value(name)).as_numpy_dtype
            for name in types])
      manifest_file.close()

    if FLAGS.tf_xla_flags is not None:
      os.environ['TF_XLA_FLAGS'] = FLAGS.tf_xla_flags

  @property
  def all_tf_types(self):
    name = '{}.{}'.format(type(self).__name__, self._testMethodName)
    tf_types = set([dtypes.as_dtype(t)
                    for t in self._method_types_filter.get(name, set())])
    return self._all_tf_types - tf_types

  @property
  def float_types(self):
    name = '{}.{}'.format(type(self).__name__, self._testMethodName)
    return self._float_types - self._method_types_filter.get(name, set())

  @property
  def float_tf_types(self):
    name = '{}.{}'.format(type(self).__name__, self._testMethodName)
    return self._float_tf_types - self._method_types_filter.get(name, set())

  @property
  def int_types(self):
    name = '{}.{}'.format(type(self).__name__, self._testMethodName)
    return self._int_types - self._method_types_filter.get(name, set())

  @property
  def numeric_tf_types(self):
    name = '{}.{}'.format(type(self).__name__, self._testMethodName)
    tf_types = set([dtypes.as_dtype(t)
                    for t in self._method_types_filter.get(name, set())])
    return self._numeric_tf_types - tf_types

  @property
  def numeric_types(self):
    name = '{}.{}'.format(type(self).__name__, self._testMethodName)
    return self._numeric_types - self._method_types_filter.get(name, set())

  @property
  def all_types(self):
    name = '{}.{}'.format(type(self).__name__, self._testMethodName)
    return self._all_types - self._method_types_filter.get(name, set())

  def setUp(self):
    super(XLATestCase, self).setUp()
    name = '{}.{}'.format(type(self).__name__, self._testMethodName)
    if self.disabled_regex is not None and self.disabled_regex.match(name):
      logging.info('Disabled test case: %s', name)
      self.skipTest('{} is disabled by manifest.'.format(name))
      return
    logging.info('Start test case: %s', name)

    random.seed(random_seed.DEFAULT_GRAPH_SEED)
    np.random.seed(random_seed.DEFAULT_GRAPH_SEED)

  def tearDown(self):
    super(XLATestCase, self).tearDown()
    logging.info('End test case: %s', self._testMethodName)

  @contextlib.contextmanager
  def test_session(self):
    """Custom implementation of test_session() for XLA tests.

    We override the standard Tensorflow test_session() since it is too
    specific to CPU and GPU tests. In particular, we want to disable soft
    placement and explicitly assign ops to devices under test.

    Yields:
      A session to use when running a test case.
    """
    graph = ops.Graph()
    with session.Session(graph=graph) as sess, graph.as_default():
      yield sess

  @contextlib.contextmanager
  def test_scope(self):
    """Test scope that runs tests on a Tensorflow/XLA device.

    Uses a compilation_scope() to mark operators to compile.

    Yields:
      A scope to apply to the operators under test.
    """
    with ops.device('device:{}:0'.format(self.device)):
      yield


def Benchmark(tf_bench,
              builder_fn,
              use_xla_jit,
              device,
              separate_compiled_gradients=False):
  """Build a graph and run benchmarks against it, with or without XLA.

  Args:
    tf_bench: An instance of tf.test.Benchmark, used to run the benchmark.
    builder_fn: A function that builds a graph when invoked, and returns
        (name, fetches), where name is the name of the test, and fetches
        is a list of tensors to fetch as output.
    use_xla_jit: If true compile with the XLA JIT, otherwise use regular TF.
    device: The tensorflow device to run on, e.g. "cpu", "gpu".
    separate_compiled_gradients: If true put each gradient subgraph into a
      separate compilation scope. This gives fine-grained control over which
      portions of the graph will be compiled as a single unit. Compiling
      gradients separately may yield better performance for some graphs.
      The scope is named based on the scope of the forward computation as well
      as the name of the gradients. As a result, the gradients will be compiled
      in a scope that is separate from both the forward computation, and from
      other gradients.
  """

  with ops.Graph().as_default():
    name = None
    targets = []
    with ops.device(device):
      fetches = []
      jit_scope = jit.experimental_jit_scope
      with jit_scope(
          compile_ops=use_xla_jit,
          separate_compiled_gradients=separate_compiled_gradients):
        name, fetches = builder_fn()

      # We only want to benchmark the operations themselves, and not the data
      # transfer of the result(s).  Non-compiled identity ops ensure XLA
      # doesn't know we're dropping the results, otherwise it might compile
      # away the entire computation.
      for fetch in fetches:
        targets.append(array_ops.identity(fetch).op)

    config = config_pb2.ConfigProto(allow_soft_placement=True)
    with session.Session(config=config) as sess:
      sess.run(variables.global_variables_initializer())
      xla = 'xla_' if use_xla_jit else ''
      tf_bench.run_op_benchmark(
          sess, targets, name='%s_%s%s' % (name, xla, device))
