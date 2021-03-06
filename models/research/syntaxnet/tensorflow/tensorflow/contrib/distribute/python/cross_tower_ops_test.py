# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for CrossTowerOps."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import parameterized

from tensorflow.contrib.distribute.python import combinations
from tensorflow.contrib.distribute.python import cross_tower_ops as cross_tower_ops_lib
from tensorflow.contrib.distribute.python import values as value_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def _make_per_device(values, devices):
  devices = cross_tower_ops_lib._get_devices_from(devices)
  assert len(values) == len(devices)
  index = {}
  for d, v in zip(devices, values):
    with ops.device(d):
      placed_v = array_ops.identity(v)
    index[d] = placed_v
  return value_lib.PerDevice(index)


# pylint: disable=g-doc-args,g-doc-return-or-yield
def _fake_mirrored(value, devices):
  """Create a faked Mirrored object for testing.

  All components of the returned Mirrored have the same objects, which is not
  true in reality.
  """
  devices = cross_tower_ops_lib._get_devices_from(devices)
  return value_lib.Mirrored(
      {d: v for d, v in zip(devices, [value] * len(devices))})


_cpu_device = "/device:CPU:0"


class CrossTowerOpsTest(test.TestCase, parameterized.TestCase):

  def _assert_value_equal(self, left, right):
    if isinstance(left, list):
      for l, r in zip(left, right):
        self._assert_value_equal(l, r)
    else:
      self.assertEqual(type(left), type(right))
      self.assertEqual(left.devices, right.devices)
      if context.executing_eagerly():
        self.assertEqual([v.numpy() for v in left._index.values()],
                         list(right._index.values()))
      else:
        with self.test_session() as sess:
          self.assertEqual(
              sess.run(list(left._index.values())), list(right._index.values()))

  # TODO(yuefengz): decouple the num_gpus check from distribution in
  # combinations module so that we can pass in devices instead of a distribution
  # strategy.
  reduction_to_one_combinations = combinations.combine(
      cross_tower_ops=[
          combinations.NamedObject(
              "DefaultReductionToOneDeviceCrossTowerOps",
              cross_tower_ops_lib.ReductionToOneDeviceCrossTowerOps()),
          combinations.NamedObject(
              "ReductionToCPUDeviceCrossTowerOps",
              cross_tower_ops_lib.ReductionToOneDeviceCrossTowerOps(
                  reduce_to_device=_cpu_device)),
          combinations.NamedObject(
              "AccumulateNCrossTowerOp",
              cross_tower_ops_lib.ReductionToOneDeviceCrossTowerOps(
                  accumulation_fn=math_ops.accumulate_n)),
      ],
      distribution=[
          combinations.one_device_strategy,
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.mirrored_strategy_with_two_gpus
      ],
      mode=["graph", "eager"])
  allreduce_combinations = combinations.combine(
      cross_tower_ops=[
          combinations.NamedObject(
              "AllReduce",
              cross_tower_ops_lib.AllReduceCrossTowerOps("nccl", 1, 0, 0)),
          combinations.NamedObject(
              "HierarchicalCopy",
              cross_tower_ops_lib.AllReduceCrossTowerOps(
                  "hierarchical_copy", 8, 0, 0)),
          combinations.NamedObject(
              "AllReduceNoGradientRepacking",
              cross_tower_ops_lib.AllReduceCrossTowerOps("nccl", 0, 0, 0)),
          combinations.NamedObject(
              "HierarchicalCopyAggregateSmallTensors",
              cross_tower_ops_lib.AllReduceCrossTowerOps(
                  "hierarchical_copy", 0, 100, 10))
      ],
      distribution=[combinations.mirrored_strategy_with_two_gpus],
      mode=["graph", "eager"])

  @combinations.generate(reduction_to_one_combinations + allreduce_combinations)
  def testReductionAndBroadcast(self, cross_tower_ops, distribution):
    devices = distribution.worker_devices

    values = [constant_op.constant(float(d)) for d in range(len(devices))]
    per_device = _make_per_device(values, devices)
    mean = (len(devices) - 1.) / 2.

    values_2 = [constant_op.constant(d + 1.0) for d in range(len(devices))]
    per_device_2 = _make_per_device(values_2, devices)
    mean_2 = mean + 1.

    destination_mirrored = _fake_mirrored(1., devices)
    destination_different = _fake_mirrored(1., _cpu_device)
    destination_str = _cpu_device
    destination_list = devices

    all_destinations = [
        None, destination_mirrored, destination_different, destination_str,
        destination_list
    ]

    # test reduce()
    for destinations in all_destinations:
      self._assert_value_equal(
          cross_tower_ops.reduce("mean", per_device, destinations=destinations),
          _fake_mirrored(mean, destinations or per_device))
      self._assert_value_equal(
          cross_tower_ops.reduce(
              "mean", per_device_2, destinations=destinations),
          _fake_mirrored(mean_2, destinations or per_device))
      self._assert_value_equal(
          cross_tower_ops.reduce("sum", per_device, destinations=destinations),
          _fake_mirrored(mean * len(devices), destinations or per_device))
      self._assert_value_equal(
          cross_tower_ops.reduce(
              "sum", per_device_2, destinations=destinations),
          _fake_mirrored(mean_2 * len(devices), destinations or per_device))

    # test batch_reduce()
    for d1, d2 in itertools.product(all_destinations, all_destinations):
      self._assert_value_equal(
          cross_tower_ops.batch_reduce(
              "mean", [(per_device, d1), (per_device_2, d2)]),
          [_fake_mirrored(mean, d1 or per_device),
           _fake_mirrored(mean_2, d2 or per_device_2)])
      self._assert_value_equal(
          cross_tower_ops.batch_reduce(
              "sum", [(per_device, d1), (per_device_2, d2)]),
          [_fake_mirrored(mean * len(devices), d1 or per_device),
           _fake_mirrored(mean_2 * len(devices), d2 or per_device_2)])

    # test broadcast()
    for destinations in all_destinations:
      if destinations is None:
        continue
      else:
        self._assert_value_equal(
            cross_tower_ops.broadcast(constant_op.constant(1.), destinations),
            _fake_mirrored(1., destinations))

  def testChooseAlgorithm(self):
    device_links = [[1, 2, 3, 4], [0, 2, 3, 5], [0, 1, 3, 6], [0, 1, 2, 7],
                    [0, 5, 6, 7], [1, 4, 6, 7], [2, 4, 5, 7], [3, 4, 5, 6]]
    result = cross_tower_ops_lib._choose_all_reduce_algorithm(device_links)
    self.assertTrue(
        isinstance(result, cross_tower_ops_lib.AllReduceCrossTowerOps))
    self.assertEqual(result.all_reduce_alg, "hierarchical_copy")
    self.assertEqual(result.num_packs, 8)

    # if there are only 4 devices
    device_links = [[1, 2, 3, 4], [0, 2, 3, 5], [0, 1, 3, 6], [0, 1, 2, 7]]
    result = cross_tower_ops_lib._choose_all_reduce_algorithm(device_links)
    self.assertTrue(
        isinstance(result, cross_tower_ops_lib.AllReduceCrossTowerOps))
    self.assertEqual(result.all_reduce_alg, "nccl")
    self.assertEqual(result.num_packs, 1)

    # if devices links contain each device itself
    device_links = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2, 3, 6],
                    [0, 1, 2, 3, 7], [0, 4, 5, 6, 7], [1, 4, 5, 6, 7],
                    [2, 4, 5, 6, 7], [3, 4, 5, 6, 7]]
    result = cross_tower_ops_lib._choose_all_reduce_algorithm(device_links)
    self.assertTrue(
        isinstance(result, cross_tower_ops_lib.AllReduceCrossTowerOps))
    self.assertEqual(result.all_reduce_alg, "hierarchical_copy")
    self.assertEqual(result.num_packs, 8)

    # if not dgx1-like links
    device_links = [[0, 2, 3, 5], [0, 1, 3, 6], [0, 1, 2, 7], [0, 5, 6, 7],
                    [1, 4, 6, 7], [2, 4, 5, 7], [3, 4, 5, 6], [1, 2, 3, 4]]
    result = cross_tower_ops_lib._choose_all_reduce_algorithm(device_links)
    self.assertTrue(
        isinstance(result, cross_tower_ops_lib.AllReduceCrossTowerOps))
    self.assertEqual(result.all_reduce_alg, "nccl")
    self.assertEqual(result.num_packs, 1)


if __name__ == "__main__":
  test.main()
