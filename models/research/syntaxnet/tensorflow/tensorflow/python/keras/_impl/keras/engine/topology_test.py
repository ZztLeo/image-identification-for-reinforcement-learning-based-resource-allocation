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
#,============================================================================
"""Tests for layer graphs construction & handling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.keras._impl import keras
from tensorflow.python.layers import base as tf_base_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import test

try:
  import yaml  # pylint:disable=g-import-not-at-top
except ImportError:
  yaml = None


class TopologyConstructionTest(test.TestCase):

  def test_get_updates(self):

    class MyLayer(keras.layers.Layer):

      def build(self, input_shape):
        self.a = self.add_variable('a',
                                   (1, 1),
                                   'float32',
                                   trainable=False)
        self.b = self.add_variable('b',
                                   (1, 1),
                                   'float32',
                                   trainable=False)
        self.add_update(state_ops.assign_add(self.a, [[1.]]))
        self.built = True

      def call(self, inputs):
        self.add_update(state_ops.assign_add(self.a, inputs),
                        inputs=True)
        return inputs + 1

    x1 = keras.Input(shape=(1,))
    layer = MyLayer()
    _ = layer.apply(x1)

    self.assertEqual(len(layer.updates), 2)
    self.assertEqual(len(layer.get_updates_for(x1)), 1)
    self.assertEqual(len(layer.get_updates_for(None)), 1)

    x2 = keras.Input(shape=(1,))
    y2 = layer.apply(x2)

    self.assertEqual(len(layer.updates), 3)
    self.assertEqual(len(layer.get_updates_for(x1)), 1)
    self.assertEqual(len(layer.get_updates_for(x2)), 1)
    self.assertEqual(len(layer.get_updates_for(None)), 1)

    network = keras.engine.Network(x2, y2)
    self.assertEqual(len(network.updates), 2)
    self.assertEqual(len(network.get_updates_for(x1)), 0)
    self.assertEqual(len(network.get_updates_for(x2)), 1)
    self.assertEqual(len(network.get_updates_for(None)), 1)

    x3 = keras.Input(shape=(1,))
    _ = layer.apply(x3)
    self.assertEqual(len(network.updates), 2)

    x4 = keras.Input(shape=(1,))
    _ = network(x4)
    self.assertEqual(len(network.updates), 3)
    self.assertEqual(len(network.get_updates_for(x2)), 1)
    self.assertEqual(len(network.get_updates_for(x4)), 1)
    self.assertEqual(len(network.get_updates_for(None)), 1)

    network.add_update(state_ops.assign_add(layer.a, [[1]]))
    self.assertEqual(len(network.updates), 4)
    self.assertEqual(len(network.get_updates_for(None)), 2)

    network.add_update(state_ops.assign_add(layer.a, x4), inputs=True)
    self.assertEqual(len(network.updates), 5)
    self.assertEqual(len(network.get_updates_for(x4)), 2)

  def test_get_losses(self):

    class MyLayer(keras.layers.Layer):

      def build(self, input_shape):
        self.a = self.add_variable('a',
                                   (1, 1),
                                   'float32',
                                   trainable=False)
        self.b = self.add_variable('b',
                                   (1, 1),
                                   'float32',
                                   trainable=False)
        self.add_loss(math_ops.reduce_sum(self.a))
        self.built = True

      def call(self, inputs):
        self.add_loss(math_ops.reduce_sum(inputs),
                      inputs=True)
        return inputs + 1

    x1 = keras.Input(shape=(1,))
    layer = MyLayer()
    _ = layer.apply(x1)

    self.assertEqual(len(layer.losses), 2)
    self.assertEqual(len(layer.get_losses_for(x1)), 1)
    self.assertEqual(len(layer.get_losses_for(None)), 1)

    x2 = keras.Input(shape=(1,))
    y2 = layer.apply(x2)

    self.assertEqual(len(layer.losses), 3)
    self.assertEqual(len(layer.get_losses_for(x1)), 1)
    self.assertEqual(len(layer.get_losses_for(x2)), 1)
    self.assertEqual(len(layer.get_losses_for(None)), 1)

    network = keras.engine.Network(x2, y2)
    self.assertEqual(len(network.losses), 2)
    self.assertEqual(len(network.get_losses_for(x1)), 0)
    self.assertEqual(len(network.get_losses_for(x2)), 1)
    self.assertEqual(len(network.get_losses_for(None)), 1)

    x3 = keras.Input(shape=(1,))
    _ = layer.apply(x3)
    self.assertEqual(len(network.losses), 2)

    x4 = keras.Input(shape=(1,))
    _ = network(x4)
    self.assertEqual(len(network.losses), 3)
    self.assertEqual(len(network.get_losses_for(x2)), 1)
    self.assertEqual(len(network.get_losses_for(x4)), 1)
    self.assertEqual(len(network.get_losses_for(None)), 1)

    network.add_loss(math_ops.reduce_sum(layer.a))
    self.assertEqual(len(network.losses), 4)
    self.assertEqual(len(network.get_losses_for(None)), 2)

    network.add_loss(math_ops.reduce_sum(x4), inputs=True)
    self.assertEqual(len(network.losses), 5)
    self.assertEqual(len(network.get_losses_for(x4)), 2)

  def testTopologicalAttributes(self):
    # test layer attributes / methods related to cross-layer connectivity.
    a = keras.Input(shape=(32,), name='input_a')
    b = keras.Input(shape=(32,), name='input_b')

    # test input, output, input_shape, output_shape
    test_layer = keras.layers.Dense(16, name='test_layer')
    a_test = test_layer(a)
    self.assertEqual(test_layer.input, a)
    self.assertEqual(test_layer.output, a_test)
    self.assertEqual(test_layer.input_shape, (None, 32))
    self.assertEqual(test_layer.output_shape, (None, 16))

    # test `get_*_at` methods
    dense = keras.layers.Dense(16, name='dense_1')
    a_2 = dense(a)
    b_2 = dense(b)

    self.assertEqual(dense.get_input_at(0), a)
    self.assertEqual(dense.get_input_at(1), b)
    self.assertEqual(dense.get_output_at(0), a_2)
    self.assertEqual(dense.get_output_at(1), b_2)
    self.assertEqual(dense.get_input_shape_at(0), (None, 32))
    self.assertEqual(dense.get_input_shape_at(1), (None, 32))
    self.assertEqual(dense.get_output_shape_at(0), (None, 16))
    self.assertEqual(dense.get_output_shape_at(1), (None, 16))

    # Test invalid value for attribute retrieval.
    with self.assertRaises(ValueError):
      dense.get_input_at(2)
    with self.assertRaises(AttributeError):
      new_dense = keras.layers.Dense(16)
      _ = new_dense.input
    with self.assertRaises(AttributeError):
      new_dense = keras.layers.Dense(16)
      _ = new_dense.output
    with self.assertRaises(AttributeError):
      new_dense = keras.layers.Dense(16)
      _ = new_dense.output_shape
    with self.assertRaises(AttributeError):
      new_dense = keras.layers.Dense(16)
      _ = new_dense.input_shape
    with self.assertRaises(AttributeError):
      new_dense = keras.layers.Dense(16)
      a = keras.Input(shape=(3, 32))
      a = keras.Input(shape=(5, 32))
      a_2 = dense(a)
      b_2 = dense(b)
      _ = new_dense.input_shape
    with self.assertRaises(AttributeError):
      new_dense = keras.layers.Dense(16)
      a = keras.Input(shape=(3, 32))
      a = keras.Input(shape=(5, 32))
      a_2 = dense(a)
      b_2 = dense(b)
      _ = new_dense.output_shape

  def testTopologicalAttributesMultiOutputLayer(self):

    class PowersLayer(keras.layers.Layer):

      def call(self, inputs):
        return [inputs**2, inputs**3]

    x = keras.Input(shape=(32,))
    test_layer = PowersLayer()
    p1, p2 = test_layer(x)  # pylint: disable=not-callable

    self.assertEqual(test_layer.input, x)
    self.assertEqual(test_layer.output, [p1, p2])
    self.assertEqual(test_layer.input_shape, (None, 32))
    self.assertEqual(test_layer.output_shape, [(None, 32), (None, 32)])

  def testTopologicalAttributesMultiInputLayer(self):

    class AddLayer(keras.layers.Layer):

      def call(self, inputs):
        assert len(inputs) == 2
        return inputs[0] + inputs[1]

    a = keras.Input(shape=(32,))
    b = keras.Input(shape=(32,))
    test_layer = AddLayer()
    y = test_layer([a, b])  # pylint: disable=not-callable

    self.assertEqual(test_layer.input, [a, b])
    self.assertEqual(test_layer.output, y)
    self.assertEqual(test_layer.input_shape, [(None, 32), (None, 32)])
    self.assertEqual(test_layer.output_shape, (None, 32))

  def testBasicNetwork(self):
    # minimum viable network
    x = keras.Input(shape=(32,))
    dense = keras.layers.Dense(2)
    y = dense(x)
    network = keras.engine.Network(x, y, name='dense_network')

    # test basic attributes
    self.assertEqual(network.name, 'dense_network')
    self.assertEqual(len(network.layers), 2)  # InputLayer + Dense
    self.assertEqual(network.layers[1], dense)
    self.assertEqual(network.weights, dense.weights)
    self.assertEqual(network.trainable_weights, dense.trainable_weights)
    self.assertEqual(network.non_trainable_weights, dense.non_trainable_weights)

    # test callability on Input
    x_2 = keras.Input(shape=(32,))
    y_2 = network(x_2)
    self.assertEqual(y_2.get_shape().as_list(), [None, 2])

    # test callability on regular tensor
    x_2 = array_ops.placeholder(dtype='float32', shape=(None, 32))
    y_2 = network(x_2)
    self.assertEqual(y_2.get_shape().as_list(), [None, 2])

    # test network `trainable` attribute
    network.trainable = False
    self.assertEqual(network.weights, dense.weights)
    self.assertEqual(network.trainable_weights, [])
    self.assertEqual(network.non_trainable_weights,
                     dense.trainable_weights + dense.non_trainable_weights)

  def test_trainable_weights(self):
    a = keras.layers.Input(shape=(2,))
    b = keras.layers.Dense(1)(a)
    model = keras.models.Model(a, b)

    weights = model.weights
    self.assertListEqual(model.trainable_weights, weights)
    self.assertListEqual(model.non_trainable_weights, [])

    model.trainable = False
    self.assertListEqual(model.trainable_weights, [])
    self.assertListEqual(model.non_trainable_weights, weights)

    model.trainable = True
    self.assertListEqual(model.trainable_weights, weights)
    self.assertListEqual(model.non_trainable_weights, [])

    model.layers[1].trainable = False
    self.assertListEqual(model.trainable_weights, [])
    self.assertListEqual(model.non_trainable_weights, weights)

    # sequential model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, input_dim=2))
    weights = model.weights

    self.assertListEqual(model.trainable_weights, weights)
    self.assertListEqual(model.non_trainable_weights, [])

    model.trainable = False
    self.assertListEqual(model.trainable_weights, [])
    self.assertListEqual(model.non_trainable_weights, weights)

    model.trainable = True
    self.assertListEqual(model.trainable_weights, weights)
    self.assertListEqual(model.non_trainable_weights, [])

    model.layers[0].trainable = False
    self.assertListEqual(model.trainable_weights, [])
    self.assertListEqual(model.non_trainable_weights, weights)

  def test_learning_phase(self):
    with self.test_session():
      a = keras.layers.Input(shape=(32,), name='input_a')
      b = keras.layers.Input(shape=(32,), name='input_b')

      a_2 = keras.layers.Dense(16, name='dense_1')(a)
      dp = keras.layers.Dropout(0.5, name='dropout')
      b_2 = dp(b)

      self.assertFalse(a_2._uses_learning_phase)
      self.assertTrue(b_2._uses_learning_phase)

      # test merge
      m = keras.layers.concatenate([a_2, b_2])
      self.assertTrue(m._uses_learning_phase)

      # Test recursion
      model = keras.models.Model([a, b], [a_2, b_2])
      self.assertTrue(model.uses_learning_phase)

      c = keras.layers.Input(shape=(32,), name='input_c')
      d = keras.layers.Input(shape=(32,), name='input_d')

      c_2, b_2 = model([c, d])
      self.assertTrue(c_2._uses_learning_phase)
      self.assertTrue(b_2._uses_learning_phase)

      # try actually running graph
      fn = keras.backend.function(
          model.inputs + [keras.backend.learning_phase()], model.outputs)
      input_a_np = np.random.random((10, 32))
      input_b_np = np.random.random((10, 32))
      fn_outputs_no_dp = fn([input_a_np, input_b_np, 0])
      fn_outputs_dp = fn([input_a_np, input_b_np, 1])
      # output a: nothing changes
      self.assertEqual(fn_outputs_no_dp[0].sum(), fn_outputs_dp[0].sum())
      # output b: dropout applied
      self.assertNotEqual(fn_outputs_no_dp[1].sum(), fn_outputs_dp[1].sum())

  def test_layer_call_arguments(self):
    # Test the ability to pass and serialize arguments to `call`.
    inp = keras.layers.Input(shape=(2,))
    x = keras.layers.Dense(3)(inp)
    x = keras.layers.Dropout(0.5)(x, training=True)
    model = keras.models.Model(inp, x)
    self.assertFalse(model.uses_learning_phase)

    # Test that argument is kept when applying the model
    inp2 = keras.layers.Input(shape=(2,))
    out2 = model(inp2)
    self.assertFalse(out2._uses_learning_phase)

    # Test that argument is kept after loading a model
    config = model.get_config()
    model = keras.models.Model.from_config(config)
    self.assertFalse(model.uses_learning_phase)

  def test_node_construction(self):
    # test basics
    a = keras.layers.Input(shape=(32,), name='input_a')
    b = keras.layers.Input(shape=(32,), name='input_b')

    with self.assertRaises(ValueError):
      _ = keras.layers.Input(shape=(32,), batch_shape=(10, 32))
    with self.assertRaises(ValueError):
      _ = keras.layers.Input(shape=(32,), unknown_kwarg=None)

    self.assertListEqual(a.get_shape().as_list(), [None, 32])
    a_layer, a_node_index, a_tensor_index = a._keras_history
    b_layer, _, _ = b._keras_history
    self.assertEqual(len(a_layer._inbound_nodes), 1)
    self.assertEqual(a_tensor_index, 0)
    node = a_layer._inbound_nodes[a_node_index]
    self.assertEqual(node.outbound_layer, a_layer)

    self.assertListEqual(node.inbound_layers, [])
    self.assertListEqual(node.input_tensors, [a])
    self.assertListEqual(node.input_shapes, [(None, 32)])
    self.assertListEqual(node.output_tensors, [a])
    self.assertListEqual(node.output_shapes, [(None, 32)])

    dense = keras.layers.Dense(16, name='dense_1')
    a_2 = dense(a)
    b_2 = dense(b)

    self.assertEqual(len(dense._inbound_nodes), 2)
    self.assertEqual(len(dense._outbound_nodes), 0)
    self.assertListEqual(dense._inbound_nodes[0].inbound_layers, [a_layer])
    self.assertEqual(dense._inbound_nodes[0].outbound_layer, dense)
    self.assertListEqual(dense._inbound_nodes[1].inbound_layers, [b_layer])
    self.assertEqual(dense._inbound_nodes[1].outbound_layer, dense)
    self.assertListEqual(dense._inbound_nodes[0].input_tensors, [a])
    self.assertListEqual(dense._inbound_nodes[1].input_tensors, [b])

    # test layer properties
    test_layer = keras.layers.Dense(16, name='test_layer')
    a_test = test_layer(a)
    self.assertListEqual(test_layer.kernel.get_shape().as_list(), [32, 16])
    self.assertEqual(test_layer.input, a)
    self.assertEqual(test_layer.output, a_test)
    self.assertEqual(test_layer.input_shape, (None, 32))
    self.assertEqual(test_layer.output_shape, (None, 16))

    self.assertEqual(dense.get_input_at(0), a)
    self.assertEqual(dense.get_input_at(1), b)
    self.assertEqual(dense.get_output_at(0), a_2)
    self.assertEqual(dense.get_output_at(1), b_2)
    self.assertEqual(dense.get_input_shape_at(0), (None, 32))
    self.assertEqual(dense.get_input_shape_at(1), (None, 32))
    self.assertEqual(dense.get_output_shape_at(0), (None, 16))
    self.assertEqual(dense.get_output_shape_at(1), (None, 16))
    self.assertEqual(dense.get_input_mask_at(0), None)
    self.assertEqual(dense.get_input_mask_at(1), None)
    self.assertEqual(dense.get_output_mask_at(0), None)
    self.assertEqual(dense.get_output_mask_at(1), None)

  def test_multi_input_layer(self):
    with self.test_session():
      # test multi-input layer
      a = keras.layers.Input(shape=(32,), name='input_a')
      b = keras.layers.Input(shape=(32,), name='input_b')

      dense = keras.layers.Dense(16, name='dense_1')
      a_2 = dense(a)
      b_2 = dense(b)

      merged = keras.layers.concatenate([a_2, b_2], name='merge')
      self.assertListEqual(merged.get_shape().as_list(), [None, 16 * 2])
      merge_layer, merge_node_index, merge_tensor_index = merged._keras_history

      self.assertEqual(merge_node_index, 0)
      self.assertEqual(merge_tensor_index, 0)

      self.assertEqual(len(merge_layer._inbound_nodes), 1)
      self.assertEqual(len(merge_layer._outbound_nodes), 0)

      self.assertEqual(len(merge_layer._inbound_nodes[0].input_tensors), 2)
      self.assertEqual(len(merge_layer._inbound_nodes[0].inbound_layers), 2)

      c = keras.layers.Dense(64, name='dense_2')(merged)
      d = keras.layers.Dense(5, name='dense_3')(c)

      model = keras.models.Model(inputs=[a, b], outputs=[c, d], name='model')
      self.assertEqual(len(model.layers), 6)
      output_shapes = model.compute_output_shape([(None, 32), (None, 32)])
      self.assertListEqual(output_shapes[0].as_list(), [None, 64])
      self.assertListEqual(output_shapes[1].as_list(), [None, 5])
      self.assertListEqual(
          model.compute_mask([a, b], [None, None]), [None, None])

      # we don't check names of first 2 layers (inputs) because
      # ordering of same-level layers is not fixed
      self.assertListEqual([l.name for l in model.layers][2:],
                           ['dense_1', 'merge', 'dense_2', 'dense_3'])
      self.assertListEqual([l.name for l in model._input_layers],
                           ['input_a', 'input_b'])
      self.assertListEqual([l.name for l in model._output_layers],
                           ['dense_2', 'dense_3'])

      # actually run model
      fn = keras.backend.function(model.inputs, model.outputs)
      input_a_np = np.random.random((10, 32))
      input_b_np = np.random.random((10, 32))
      fn_outputs = fn([input_a_np, input_b_np])
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 64), (10, 5)])

      # test get_source_inputs
      self.assertListEqual(keras.engine.network.get_source_inputs(c), [a, b])

      # serialization / deserialization
      json_config = model.to_json()
      recreated_model = keras.models.model_from_json(json_config)
      recreated_model.compile('rmsprop', 'mse')

      self.assertListEqual([l.name for l in recreated_model.layers][2:],
                           ['dense_1', 'merge', 'dense_2', 'dense_3'])
      self.assertListEqual([l.name for l in recreated_model._input_layers],
                           ['input_a', 'input_b'])
      self.assertListEqual([l.name for l in recreated_model._output_layers],
                           ['dense_2', 'dense_3'])

      fn = keras.backend.function(recreated_model.inputs,
                                  recreated_model.outputs)
      input_a_np = np.random.random((10, 32))
      input_b_np = np.random.random((10, 32))
      fn_outputs = fn([input_a_np, input_b_np])
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 64), (10, 5)])

  def test_recursion(self):
    with self.test_session():
      a = keras.layers.Input(shape=(32,), name='input_a')
      b = keras.layers.Input(shape=(32,), name='input_b')

      dense = keras.layers.Dense(16, name='dense_1')
      a_2 = dense(a)
      b_2 = dense(b)
      merged = keras.layers.concatenate([a_2, b_2], name='merge')
      c = keras.layers.Dense(64, name='dense_2')(merged)
      d = keras.layers.Dense(5, name='dense_3')(c)

      model = keras.models.Model(inputs=[a, b], outputs=[c, d], name='model')

      e = keras.layers.Input(shape=(32,), name='input_e')
      f = keras.layers.Input(shape=(32,), name='input_f')
      self.assertEqual(len(model.inputs), 2)
      g, h = model([e, f])
      self.assertEqual(len(model.inputs), 2)
      self.assertEqual(g.name, 'model/dense_2/BiasAdd:0')

      self.assertListEqual(g.get_shape().as_list(), c.get_shape().as_list())
      self.assertListEqual(h.get_shape().as_list(), d.get_shape().as_list())

      # test separate manipulation of different layer outputs
      i = keras.layers.Dense(7, name='dense_4')(h)

      final_model = keras.models.Model(
          inputs=[e, f], outputs=[i, g], name='final')
      self.assertEqual(len(final_model.inputs), 2)
      self.assertEqual(len(final_model.outputs), 2)
      self.assertEqual(len(final_model.layers), 4)

      # we don't check names of first 2 layers (inputs) because
      # ordering of same-level layers is not fixed
      self.assertListEqual([layer.name for layer in final_model.layers][2:],
                           ['model', 'dense_4'])
      self.assertListEqual(
          model.compute_mask([e, f], [None, None]), [None, None])
      self.assertListEqual(
          final_model.compute_output_shape([(10, 32), (10, 32)]), [(10, 7),
                                                                   (10, 64)])

      # run recursive model
      fn = keras.backend.function(final_model.inputs, final_model.outputs)
      input_a_np = np.random.random((10, 32))
      input_b_np = np.random.random((10, 32))
      fn_outputs = fn([input_a_np, input_b_np])
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 7), (10, 64)])

      # test serialization
      model_config = final_model.get_config()
      recreated_model = keras.models.Model.from_config(model_config)

      fn = keras.backend.function(recreated_model.inputs,
                                  recreated_model.outputs)
      input_a_np = np.random.random((10, 32))
      input_b_np = np.random.random((10, 32))
      fn_outputs = fn([input_a_np, input_b_np])
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 7), (10, 64)])

  def test_multi_input_multi_output_recursion(self):
    with self.test_session():
      # test multi-input multi-output
      a = keras.layers.Input(shape=(32,), name='input_a')
      b = keras.layers.Input(shape=(32,), name='input_b')

      dense = keras.layers.Dense(16, name='dense_1')
      a_2 = dense(a)
      b_2 = dense(b)
      merged = keras.layers.concatenate([a_2, b_2], name='merge')
      c = keras.layers.Dense(64, name='dense_2')(merged)
      d = keras.layers.Dense(5, name='dense_3')(c)

      model = keras.models.Model(inputs=[a, b], outputs=[c, d], name='model')

      j = keras.layers.Input(shape=(32,), name='input_j')
      k = keras.layers.Input(shape=(32,), name='input_k')
      _, n = model([j, k])

      o = keras.layers.Input(shape=(32,), name='input_o')
      p = keras.layers.Input(shape=(32,), name='input_p')
      q, _ = model([o, p])

      self.assertListEqual(n.get_shape().as_list(), [None, 5])
      self.assertListEqual(q.get_shape().as_list(), [None, 64])
      s = keras.layers.concatenate([n, q], name='merge_nq')
      self.assertListEqual(s.get_shape().as_list(), [None, 64 + 5])

      # test with single output as 1-elem list
      multi_io_model = keras.models.Model([j, k, o, p], [s])

      fn = keras.backend.function(multi_io_model.inputs, multi_io_model.outputs)
      fn_outputs = fn([
          np.random.random((10, 32)), np.random.random((10, 32)),
          np.random.random((10, 32)), np.random.random((10, 32))
      ])
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 69)])

      # test with single output as tensor
      multi_io_model = keras.models.Model([j, k, o, p], s)

      fn = keras.backend.function(multi_io_model.inputs, multi_io_model.outputs)
      fn_outputs = fn([
          np.random.random((10, 32)), np.random.random((10, 32)),
          np.random.random((10, 32)), np.random.random((10, 32))
      ])
      # note that the output of the function will still be a 1-elem list
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 69)])

      # test serialization
      model_config = multi_io_model.get_config()
      recreated_model = keras.models.Model.from_config(model_config)

      fn = keras.backend.function(recreated_model.inputs,
                                  recreated_model.outputs)
      fn_outputs = fn([
          np.random.random((10, 32)), np.random.random((10, 32)),
          np.random.random((10, 32)), np.random.random((10, 32))
      ])
      # note that the output of the function will still be a 1-elem list
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 69)])

      config = model.get_config()
      keras.models.Model.from_config(config)

      model.summary()
      json_str = model.to_json()
      keras.models.model_from_json(json_str)

      if yaml is not None:
        yaml_str = model.to_yaml()
        keras.models.model_from_yaml(yaml_str)

  def test_invalid_graphs(self):
    a = keras.layers.Input(shape=(32,), name='input_a')
    b = keras.layers.Input(shape=(32,), name='input_b')

    dense = keras.layers.Dense(16, name='dense_1')
    a_2 = dense(a)
    b_2 = dense(b)
    merged = keras.layers.concatenate([a_2, b_2], name='merge')
    c = keras.layers.Dense(64, name='dense_2')(merged)
    d = keras.layers.Dense(5, name='dense_3')(c)

    model = keras.models.Model(inputs=[a, b], outputs=[c, d], name='model')

    # input is not an Input tensor
    j = keras.layers.Input(shape=(32,), name='input_j')
    j = keras.layers.Dense(32)(j)
    k = keras.layers.Input(shape=(32,), name='input_k')
    m, n = model([j, k])

    with self.assertRaises(Exception):
      keras.models.Model([j, k], [m, n])

    # disconnected graph
    j = keras.layers.Input(shape=(32,), name='input_j')
    k = keras.layers.Input(shape=(32,), name='input_k')
    m, n = model([j, k])
    with self.assertRaises(Exception):
      keras.models.Model([j], [m, n])

    # redundant outputs
    j = keras.layers.Input(shape=(32,), name='input_j')
    k = keras.layers.Input(shape=(32,), name='input_k')
    m, n = model([j, k])

    keras.models.Model([j, k], [m, n, n])

    # redundant inputs
    j = keras.layers.Input(shape=(32,), name='input_j')
    k = keras.layers.Input(shape=(32,), name='input_k')
    m, n = model([j, k])
    with self.assertRaises(Exception):
      keras.models.Model([j, k, j], [m, n])

    # i have not idea what I'm doing: garbage as inputs/outputs
    j = keras.layers.Input(shape=(32,), name='input_j')
    k = keras.layers.Input(shape=(32,), name='input_k')
    m, n = model([j, k])
    with self.assertRaises(Exception):
      keras.models.Model([j, k], [m, n, 0])

  def test_raw_tf_compatibility(self):
    # test calling layers/models on TF tensors
    a = keras.layers.Input(shape=(32,), name='input_a')
    b = keras.layers.Input(shape=(32,), name='input_b')

    dense = keras.layers.Dense(16, name='dense_1')
    a_2 = dense(a)
    b_2 = dense(b)
    merged = keras.layers.concatenate([a_2, b_2], name='merge')
    c = keras.layers.Dense(64, name='dense_2')(merged)
    d = keras.layers.Dense(5, name='dense_3')(c)

    model = keras.models.Model(inputs=[a, b], outputs=[c, d], name='model')

    j = keras.layers.Input(shape=(32,), name='input_j')
    k = keras.layers.Input(shape=(32,), name='input_k')
    self.assertEqual(len(model.inputs), 2)
    m, n = model([j, k])
    self.assertEqual(len(model.inputs), 2)
    tf_model = keras.models.Model([j, k], [m, n])

    j_tf = array_ops.placeholder(dtype=dtypes.float32, shape=(None, 32))
    k_tf = array_ops.placeholder(dtype=dtypes.float32, shape=(None, 32))
    m_tf, n_tf = tf_model([j_tf, k_tf])
    self.assertListEqual(m_tf.get_shape().as_list(), [None, 64])
    self.assertListEqual(n_tf.get_shape().as_list(), [None, 5])

    # test merge
    keras.layers.concatenate([j_tf, k_tf], axis=1)
    keras.layers.add([j_tf, k_tf])

    # test tensor input
    x = array_ops.placeholder(shape=(None, 2), dtype=dtypes.float32)
    keras.layers.InputLayer(input_tensor=x)

    x = keras.layers.Input(tensor=x)
    keras.layers.Dense(2)(x)

  def test_basic_masking(self):
    a = keras.layers.Input(shape=(10, 32), name='input_a')
    b = keras.layers.Masking()(a)
    model = keras.models.Model(a, b)
    self.assertEqual(model.output_mask.get_shape().as_list(), [None, 10])

  def testMaskingSingleInput(self):

    class MaskedLayer(keras.layers.Layer):

      def call(self, inputs, mask=None):
        if mask is not None:
          return inputs * mask
        return inputs

      def compute_mask(self, inputs, mask=None):
        return array_ops.ones_like(inputs)

    if context.executing_eagerly():
      a = constant_op.constant([2] * 32)
      mask = constant_op.constant([0, 1] * 16)
      a._keras_mask = mask
      b = MaskedLayer().apply(a)
      self.assertTrue(hasattr(b, '_keras_mask'))
      self.assertAllEqual(
          self.evaluate(array_ops.ones_like(mask)),
          self.evaluate(getattr(b, '_keras_mask')))
      self.assertAllEqual(self.evaluate(a * mask), self.evaluate(b))
    else:
      x = keras.Input(shape=(32,))
      y = MaskedLayer()(x)  # pylint: disable=not-callable
      network = keras.engine.Network(x, y)

      # test callability on Input
      x_2 = keras.Input(shape=(32,))
      y_2 = network(x_2)
      self.assertEqual(y_2.get_shape().as_list(), [None, 32])

      # test callability on regular tensor
      x_2 = array_ops.placeholder(dtype='float32', shape=(None, 32))
      y_2 = network(x_2)
      self.assertEqual(y_2.get_shape().as_list(), [None, 32])

  def test_activity_regularization_with_model_composition(self):

    def reg(x):
      return math_ops.reduce_sum(x)

    net_a_input = keras.Input((2,))
    net_a = net_a_input
    net_a = keras.layers.Dense(2, kernel_initializer='ones',
                               use_bias=False,
                               activity_regularizer=reg)(net_a)
    model_a = keras.Model([net_a_input], [net_a])

    net_b_input = keras.Input((2,))
    net_b = model_a(net_b_input)
    model_b = keras.Model([net_b_input], [net_b])

    model_b.compile(optimizer='sgd', loss=None)
    x = np.ones((1, 2))
    loss = model_b.evaluate(x)
    self.assertEqual(loss, 4.)

  def test_layer_sharing_at_heterogenous_depth(self):
    with self.test_session():
      x_val = np.random.random((10, 5))

      x = keras.Input(shape=(5,))
      a = keras.layers.Dense(5, name='A')
      b = keras.layers.Dense(5, name='B')
      output = a(b(a(b(x))))
      m = keras.models.Model(x, output)

      output_val = m.predict(x_val)

      config = m.get_config()
      weights = m.get_weights()

      m2 = keras.models.Model.from_config(config)
      m2.set_weights(weights)

      output_val_2 = m2.predict(x_val)
      self.assertAllClose(output_val, output_val_2, atol=1e-6)

  def test_layer_sharing_at_heterogenous_depth_with_concat(self):
    with self.test_session():
      input_shape = (16, 9, 3)
      input_layer = keras.Input(shape=input_shape)

      a = keras.layers.Dense(3, name='dense_A')
      b = keras.layers.Dense(3, name='dense_B')
      c = keras.layers.Dense(3, name='dense_C')

      x1 = b(a(input_layer))
      x2 = a(c(input_layer))
      output = keras.layers.concatenate([x1, x2])

      m = keras.models.Model(inputs=input_layer, outputs=output)

      x_val = np.random.random((10, 16, 9, 3))
      output_val = m.predict(x_val)

      config = m.get_config()
      weights = m.get_weights()

      m2 = keras.models.Model.from_config(config)
      m2.set_weights(weights)

      output_val_2 = m2.predict(x_val)
      self.assertAllClose(output_val, output_val_2, atol=1e-6)

  def test_explicit_training_argument(self):
    with self.test_session():
      a = keras.layers.Input(shape=(2,))
      b = keras.layers.Dropout(0.5)(a)
      base_model = keras.models.Model(a, b)

      a = keras.layers.Input(shape=(2,))
      b = base_model(a, training=False)
      model = keras.models.Model(a, b)

      x = np.ones((100, 2))
      y = np.ones((100, 2))
      model.compile(optimizer='sgd', loss='mse')
      loss = model.train_on_batch(x, y)
      self.assertEqual(loss, 0)  # In inference mode, output is equal to input.

      a = keras.layers.Input(shape=(2,))
      b = base_model(a, training=True)
      model = keras.models.Model(a, b)
      preds = model.predict(x)
      self.assertEqual(np.min(preds), 0.)  # At least one unit was dropped.


class DeferredModeTest(test.TestCase):

  def testDeferredTensorAttributes(self):
    x = tf_base_layers._DeferredTensor(shape=(None, 2),
                                       dtype='float32',
                                       name='x')
    self.assertEqual(str(x),
                     'DeferredTensor(\'x\', shape=(?, 2), dtype=float32)')
    self.assertEqual(repr(x),
                     '<_DeferredTensor \'x\' shape=(?, 2) dtype=float32>')

  @test_util.run_in_graph_and_eager_modes()
  def testSimpleNetworkBuilding(self):
    inputs = keras.engine.Input(shape=(32,))
    if context.executing_eagerly():
      self.assertIsInstance(inputs, tf_base_layers._DeferredTensor)
      self.assertEqual(inputs.dtype.name, 'float32')
      self.assertEqual(inputs.shape.as_list(), [None, 32])

    x = keras.layers.Dense(2)(inputs)
    if context.executing_eagerly():
      self.assertIsInstance(x, tf_base_layers._DeferredTensor)
      self.assertEqual(x.dtype.name, 'float32')
      self.assertEqual(x.shape.as_list(), [None, 2])

    outputs = keras.layers.Dense(4)(x)
    network = keras.engine.Network(inputs, outputs)
    self.assertIsInstance(network, keras.engine.Network)

    if context.executing_eagerly():
      # It should be possible to call such a network on EagerTensors.
      inputs = constant_op.constant(
          np.random.random((10, 32)).astype('float32'))
      outputs = network(inputs)
      self.assertEqual(outputs.shape.as_list(), [10, 4])

  @test_util.run_in_graph_and_eager_modes()
  def testMultiIONetworkbuilding(self):
    input_a = keras.engine.Input(shape=(32,))
    input_b = keras.engine.Input(shape=(16,))
    a = keras.layers.Dense(16)(input_a)

    class AddLayer(keras.layers.Layer):

      def call(self, inputs):
        return inputs[0] + inputs[1]

      def compute_output_shape(self, input_shape):
        return input_shape[0]

    c = AddLayer()([a, input_b])  # pylint: disable=not-callable
    c = keras.layers.Dense(2)(c)

    network = keras.engine.Network([input_a, input_b], [a, c])
    if context.executing_eagerly():
      a_val = constant_op.constant(
          np.random.random((10, 32)).astype('float32'))
      b_val = constant_op.constant(
          np.random.random((10, 16)).astype('float32'))
      outputs = network([a_val, b_val])
      self.assertEqual(len(outputs), 2)
      self.assertEqual(outputs[0].shape.as_list(), [10, 16])
      self.assertEqual(outputs[1].shape.as_list(), [10, 2])

if __name__ == '__main__':
  test.main()
