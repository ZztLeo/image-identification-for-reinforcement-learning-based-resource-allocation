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
"""Tests for tensorflow.ops.data_flow_ops.FIFOQueue."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class FIFOQueueTest(test.TestCase):

  def testConstructor(self):
    with ops.Graph().as_default():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, name="Q")
    self.assertTrue(isinstance(q.queue_ref, ops.Tensor))
    self.assertProtoEquals("""
      name:'Q' op:'FIFOQueueV2'
      attr { key: 'component_types' value { list { type: DT_FLOAT } } }
      attr { key: 'shapes' value { list {} } }
      attr { key: 'capacity' value { i: 10 } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: '' } }
      """, q.queue_ref.op.node_def)

  def testMultiQueueConstructor(self):
    with ops.Graph().as_default():
      q = data_flow_ops.FIFOQueue(
          5, (dtypes_lib.int32, dtypes_lib.float32),
          shared_name="foo",
          name="Q")
    self.assertTrue(isinstance(q.queue_ref, ops.Tensor))
    self.assertProtoEquals("""
      name:'Q' op:'FIFOQueueV2'
      attr { key: 'component_types' value { list {
        type: DT_INT32 type : DT_FLOAT
      } } }
      attr { key: 'shapes' value { list {} } }
      attr { key: 'capacity' value { i: 5 } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: 'foo' } }
      """, q.queue_ref.op.node_def)

  def testConstructorWithShapes(self):
    with ops.Graph().as_default():
      q = data_flow_ops.FIFOQueue(
          5, (dtypes_lib.int32, dtypes_lib.float32),
          shapes=(tensor_shape.TensorShape([1, 1, 2, 3]),
                  tensor_shape.TensorShape([5, 8])),
          name="Q")
    self.assertTrue(isinstance(q.queue_ref, ops.Tensor))
    self.assertProtoEquals("""
      name:'Q' op:'FIFOQueueV2'
      attr { key: 'component_types' value { list {
        type: DT_INT32 type : DT_FLOAT
      } } }
      attr { key: 'shapes' value { list {
        shape { dim { size: 1 }
                dim { size: 1 }
                dim { size: 2 }
                dim { size: 3 } }
        shape { dim { size: 5 }
                dim { size: 8 } }
      } } }
      attr { key: 'capacity' value { i: 5 } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: '' } }
      """, q.queue_ref.op.node_def)

  def testEnqueue(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      enqueue_op = q.enqueue((10.0,))
      enqueue_op.run()

  def testEnqueueHalf(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float16)
      enqueue_op = q.enqueue((10.0,))
      enqueue_op.run()

  def testEnqueueWithShape(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shapes=(3, 2))
      enqueue_correct_op = q.enqueue(([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],))
      enqueue_correct_op.run()
      with self.assertRaises(ValueError):
        q.enqueue(([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],))
      self.assertEqual(1, q.size().eval())

  def testEnqueueManyWithShape(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(
          10, [dtypes_lib.int32, dtypes_lib.int32], shapes=[(), (2,)])
      q.enqueue_many([[1, 2, 3, 4], [[1, 1], [2, 2], [3, 3], [4, 4]]]).run()
      self.assertEqual(4, q.size().eval())

  def testMultipleDequeues(self):
    with self.test_session() as session:
      q = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
      q.enqueue_many([[1, 2, 3]]).run()
      a, b, c = session.run([q.dequeue(), q.dequeue(), q.dequeue()])
      self.assertAllEqual(set([1, 2, 3]), set([a, b, c]))

  def testEnqueueDictWithoutNames(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      with self.assertRaisesRegexp(ValueError, "must have names"):
        q.enqueue({"a": 12.0})
      with self.assertRaisesRegexp(ValueError, "must have names"):
        q.enqueue_many({"a": [12.0, 13.0]})

  def testParallelEnqueue(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      # Run one producer thread for each element in elems.
      def enqueue(enqueue_op):
        sess.run(enqueue_op)

      threads = [
          self.checkedThread(
              target=enqueue, args=(e,)) for e in enqueue_ops
      ]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()

      # Dequeue every element using a single thread.
      results = []
      for _ in xrange(len(elems)):
        results.append(dequeued_t.eval())
      self.assertItemsEqual(elems, results)

  def testParallelDequeue(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      # Enqueue every element using a single thread.
      for enqueue_op in enqueue_ops:
        enqueue_op.run()

      # Run one consumer thread for each element in elems.
      results = []

      def dequeue():
        results.append(sess.run(dequeued_t))

      threads = [self.checkedThread(target=dequeue) for _ in enqueue_ops]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, results)

  def testDequeue(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      for enqueue_op in enqueue_ops:
        enqueue_op.run()

      for i in xrange(len(elems)):
        vals = dequeued_t.eval()
        self.assertEqual([elems[i]], vals)

  def testDequeueHalf(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float16)
      elems = [10.0, 20.0, 30.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      for enqueue_op in enqueue_ops:
        enqueue_op.run()

      for i in xrange(len(elems)):
        vals = dequeued_t.eval()
        self.assertEqual([elems[i]], vals)

  def testEnqueueAndBlockingDequeue(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(3, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      def enqueue():
        # The enqueue_ops should run after the dequeue op has blocked.
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        for enqueue_op in enqueue_ops:
          sess.run(enqueue_op)

      results = []

      def dequeue():
        for _ in xrange(len(elems)):
          results.append(sess.run(dequeued_t))

      enqueue_thread = self.checkedThread(target=enqueue)
      dequeue_thread = self.checkedThread(target=dequeue)
      enqueue_thread.start()
      dequeue_thread.start()
      enqueue_thread.join()
      dequeue_thread.join()

      for elem, result in zip(elems, results):
        self.assertEqual([elem], result)

  def testMultiEnqueueAndDequeue(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(10, (dtypes_lib.int32, dtypes_lib.float32))
      elems = [(5, 10.0), (10, 20.0), (15, 30.0)]
      enqueue_ops = [q.enqueue((x, y)) for x, y in elems]
      dequeued_t = q.dequeue()

      for enqueue_op in enqueue_ops:
        enqueue_op.run()

      for i in xrange(len(elems)):
        x_val, y_val = sess.run(dequeued_t)
        x, y = elems[i]
        self.assertEqual([x], x_val)
        self.assertEqual([y], y_val)

  def testQueueSizeEmpty(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      self.assertEqual([0], q.size().eval())

  def testQueueSizeAfterEnqueueAndDequeue(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      enqueue_op = q.enqueue((10.0,))
      dequeued_t = q.dequeue()
      size = q.size()
      self.assertEqual([], size.get_shape())

      enqueue_op.run()
      self.assertEqual(1, size.eval())
      dequeued_t.op.run()
      self.assertEqual(0, size.eval())

  def testEnqueueMany(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue()
      enqueue_op.run()
      enqueue_op.run()

      for i in range(8):
        vals = dequeued_t.eval()
        self.assertEqual([elems[i % 4]], vals)

  def testEmptyEnqueueMany(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      empty_t = constant_op.constant(
          [], dtype=dtypes_lib.float32, shape=[0, 2, 3])
      enqueue_op = q.enqueue_many((empty_t,))
      size_t = q.size()

      self.assertEqual([0], size_t.eval())
      enqueue_op.run()
      self.assertEqual([0], size_t.eval())

  def testEmptyDequeueMany(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shapes=())
      enqueue_op = q.enqueue((10.0,))
      dequeued_t = q.dequeue_many(0)

      self.assertEqual([], dequeued_t.eval().tolist())
      enqueue_op.run()
      self.assertEqual([], dequeued_t.eval().tolist())

  def testEmptyDequeueUpTo(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shapes=())
      enqueue_op = q.enqueue((10.0,))
      dequeued_t = q.dequeue_up_to(0)

      self.assertEqual([], dequeued_t.eval().tolist())
      enqueue_op.run()
      self.assertEqual([], dequeued_t.eval().tolist())

  def testEmptyDequeueManyWithNoShape(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      # Expect the operation to fail due to the shape not being constrained.
      with self.assertRaisesOpError("specified shapes"):
        q.dequeue_many(0).eval()

  def testMultiEnqueueMany(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(10, (dtypes_lib.float32, dtypes_lib.int32))
      float_elems = [10.0, 20.0, 30.0, 40.0]
      int_elems = [[1, 2], [3, 4], [5, 6], [7, 8]]
      enqueue_op = q.enqueue_many((float_elems, int_elems))
      dequeued_t = q.dequeue()

      enqueue_op.run()
      enqueue_op.run()

      for i in range(8):
        float_val, int_val = sess.run(dequeued_t)
        self.assertEqual(float_elems[i % 4], float_val)
        self.assertAllEqual(int_elems[i % 4], int_val)

  def testDequeueMany(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, ())
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(4)

      enqueue_op.run()

      self.assertAllEqual(elems[0:4], dequeued_t.eval())
      self.assertAllEqual(elems[4:8], dequeued_t.eval())

  def testDequeueUpToNoBlocking(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, ())
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_up_to(4)

      enqueue_op.run()

      self.assertAllEqual(elems[0:4], dequeued_t.eval())
      self.assertAllEqual(elems[4:8], dequeued_t.eval())

  def testMultiDequeueMany(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(
          10, (dtypes_lib.float32, dtypes_lib.int32), shapes=((), (2,)))
      float_elems = [
          10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
      ]
      int_elems = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                   [15, 16], [17, 18], [19, 20]]
      enqueue_op = q.enqueue_many((float_elems, int_elems))
      dequeued_t = q.dequeue_many(4)
      dequeued_single_t = q.dequeue()

      enqueue_op.run()

      float_val, int_val = sess.run(dequeued_t)
      self.assertAllEqual(float_elems[0:4], float_val)
      self.assertAllEqual(int_elems[0:4], int_val)
      self.assertEqual(float_val.shape, dequeued_t[0].get_shape())
      self.assertEqual(int_val.shape, dequeued_t[1].get_shape())

      float_val, int_val = sess.run(dequeued_t)
      self.assertAllEqual(float_elems[4:8], float_val)
      self.assertAllEqual(int_elems[4:8], int_val)

      float_val, int_val = sess.run(dequeued_single_t)
      self.assertAllEqual(float_elems[8], float_val)
      self.assertAllEqual(int_elems[8], int_val)
      self.assertEqual(float_val.shape, dequeued_single_t[0].get_shape())
      self.assertEqual(int_val.shape, dequeued_single_t[1].get_shape())

  def testMultiDequeueUpToNoBlocking(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(
          10, (dtypes_lib.float32, dtypes_lib.int32), shapes=((), (2,)))
      float_elems = [
          10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
      ]
      int_elems = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                   [15, 16], [17, 18], [19, 20]]
      enqueue_op = q.enqueue_many((float_elems, int_elems))
      dequeued_t = q.dequeue_up_to(4)

      enqueue_op.run()

      float_val, int_val = sess.run(dequeued_t)
      self.assertAllEqual(float_elems[0:4], float_val)
      self.assertAllEqual(int_elems[0:4], int_val)
      self.assertEqual([None], dequeued_t[0].get_shape().as_list())
      self.assertEqual([None, 2], dequeued_t[1].get_shape().as_list())

      float_val, int_val = sess.run(dequeued_t)
      self.assertAllEqual(float_elems[4:8], float_val)
      self.assertAllEqual(int_elems[4:8], int_val)

  def testHighDimension(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.int32, (4, 4, 4, 4))
      elems = np.array([[[[[x] * 4] * 4] * 4] * 4 for x in range(10)], np.int32)
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(10)

      enqueue_op.run()
      self.assertAllEqual(dequeued_t.eval(), elems)

  def testEnqueueWrongShape(self):
    q = data_flow_ops.FIFOQueue(10, (dtypes_lib.int32, dtypes_lib.int32), ((),
                                                                           (2)))

    with self.assertRaises(ValueError):
      q.enqueue(([1, 2], [2, 2]))

    with self.assertRaises(ValueError):
      q.enqueue_many((7, [[1, 2], [3, 4], [5, 6]]))

  def testBatchSizeMismatch(self):
    q = data_flow_ops.FIFOQueue(10, (dtypes_lib.int32, dtypes_lib.int32,
                                     dtypes_lib.int32), ((), (), ()))

    with self.assertRaises(ValueError):
      q.enqueue_many(([1, 2, 3], [1, 2], [1, 2, 3]))

    with self.assertRaises(ValueError):
      q.enqueue_many(
          ([1, 2, 3], [1, 2], array_ops.placeholder(dtypes_lib.int32)))

    with self.assertRaises(ValueError):
      q.enqueue_many(
          (array_ops.placeholder(dtypes_lib.int32), [1, 2], [1, 2, 3]))

  def testEnqueueManyEmptyTypeConversion(self):
    q = data_flow_ops.FIFOQueue(10, (dtypes_lib.int32, dtypes_lib.float32), (
        (), ()))
    enq = q.enqueue_many(([], []))
    self.assertEqual(dtypes_lib.int32, enq.inputs[1].dtype)
    self.assertEqual(dtypes_lib.float32, enq.inputs[2].dtype)

  def testEnqueueWrongType(self):
    q = data_flow_ops.FIFOQueue(10, (dtypes_lib.int32, dtypes_lib.float32), (
        (), ()))

    with self.assertRaises(ValueError):
      q.enqueue((array_ops.placeholder(dtypes_lib.int32),
                 array_ops.placeholder(dtypes_lib.int32)))

    with self.assertRaises(ValueError):
      q.enqueue_many((array_ops.placeholder(dtypes_lib.int32),
                      array_ops.placeholder(dtypes_lib.int32)))

  def testEnqueueWrongShapeAtRuntime(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(10, (dtypes_lib.int32, dtypes_lib.int32), (
          (2, 2), (3, 3)))
      elems_ok = np.array([1] * 4).reshape((2, 2)).astype(np.int32)
      elems_bad = array_ops.placeholder(dtypes_lib.int32)
      enqueue_op = q.enqueue((elems_ok, elems_bad))
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   r"Expected \[3,3\], got \[3,4\]"):
        sess.run([enqueue_op],
                 feed_dict={elems_bad: np.array([1] * 12).reshape((3, 4))})

  def testEnqueueDequeueManyWrongShape(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(10, (dtypes_lib.int32, dtypes_lib.int32), (
          (2, 2), (3, 3)))
      elems_ok = np.array([1] * 8).reshape((2, 2, 2)).astype(np.int32)
      elems_bad = array_ops.placeholder(dtypes_lib.int32)
      enqueue_op = q.enqueue_many((elems_ok, elems_bad))
      dequeued_t = q.dequeue_many(2)
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Shape mismatch in tuple component 1. "
                                   r"Expected \[2,3,3\], got \[2,3,4\]"):
        sess.run([enqueue_op],
                 feed_dict={elems_bad: np.array([1] * 24).reshape((2, 3, 4))})
        dequeued_t.eval()

  def testParallelEnqueueMany(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(1000, dtypes_lib.float32, shapes=())
      elems = [10.0 * x for x in range(100)]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(1000)

      # Enqueue 100 items in parallel on 10 threads.
      def enqueue():
        sess.run(enqueue_op)

      threads = [self.checkedThread(target=enqueue) for _ in range(10)]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()

      self.assertItemsEqual(dequeued_t.eval(), elems * 10)

  def testParallelDequeueMany(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(1000, dtypes_lib.float32, shapes=())
      elems = [10.0 * x for x in range(1000)]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(100)

      enqueue_op.run()

      # Dequeue 100 items in parallel on 10 threads.
      dequeued_elems = []

      def dequeue():
        dequeued_elems.extend(sess.run(dequeued_t))

      threads = [self.checkedThread(target=dequeue) for _ in range(10)]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, dequeued_elems)

  def testParallelDequeueUpTo(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(1000, dtypes_lib.float32, shapes=())
      elems = [10.0 * x for x in range(1000)]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_up_to(101)

      enqueue_op.run()
      close_op.run()

      # Dequeue up to 101 items in parallel on 10 threads, from closed queue.
      dequeued_elems = []

      def dequeue():
        dequeued_elems.extend(sess.run(dequeued_t))

      threads = [self.checkedThread(target=dequeue) for _ in range(10)]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, dequeued_elems)

  def testParallelEnqueueAndDequeue(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(50, dtypes_lib.float32, shapes=())
      initial_elements = [10.0] * 49
      q.enqueue_many((initial_elements,)).run()

      enqueue_op = q.enqueue((20.0,))
      dequeued_t = q.dequeue()

      def enqueue():
        for _ in xrange(100):
          sess.run(enqueue_op)

      def dequeue():
        for _ in xrange(100):
          self.assertTrue(sess.run(dequeued_t) in (10.0, 20.0))

      enqueue_threads = [self.checkedThread(target=enqueue) for _ in range(10)]
      dequeue_threads = [self.checkedThread(target=dequeue) for _ in range(10)]
      for enqueue_thread in enqueue_threads:
        enqueue_thread.start()
      for dequeue_thread in dequeue_threads:
        dequeue_thread.start()
      for enqueue_thread in enqueue_threads:
        enqueue_thread.join()
      for dequeue_thread in dequeue_threads:
        dequeue_thread.join()

      # Dequeue the initial count of elements to clean up.
      cleanup_elems = q.dequeue_many(49).eval()
      for elem in cleanup_elems:
        self.assertTrue(elem in (10.0, 20.0))

  def testMixtureOfEnqueueAndEnqueueMany(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.int32, shapes=())
      enqueue_placeholder = array_ops.placeholder(dtypes_lib.int32, shape=())
      enqueue_op = q.enqueue((enqueue_placeholder,))
      enqueuemany_placeholder = array_ops.placeholder(
          dtypes_lib.int32, shape=(None,))
      enqueuemany_op = q.enqueue_many((enqueuemany_placeholder,))

      dequeued_t = q.dequeue()
      close_op = q.close()

      def dequeue():
        for i in xrange(250):
          self.assertEqual(i, sess.run(dequeued_t))

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()

      elements_enqueued = 0
      while elements_enqueued < 250:
        # With equal probability, run Enqueue or enqueue_many.
        if random.random() > 0.5:
          enqueue_op.run({enqueue_placeholder: elements_enqueued})
          elements_enqueued += 1
        else:
          count = random.randint(0, min(20, 250 - elements_enqueued))
          range_to_enqueue = np.arange(
              elements_enqueued, elements_enqueued + count, dtype=np.int32)
          enqueuemany_op.run({enqueuemany_placeholder: range_to_enqueue})
          elements_enqueued += count

      close_op.run()
      dequeue_thread.join()
      self.assertEqual(0, q.size().eval())

  def testMixtureOfDequeueAndDequeueMany(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.int32, shapes=())
      enqueue_op = q.enqueue_many((np.arange(250, dtype=np.int32),))
      dequeued_t = q.dequeue()
      count_placeholder = array_ops.placeholder(dtypes_lib.int32, shape=())
      dequeuemany_t = q.dequeue_many(count_placeholder)

      def enqueue():
        sess.run(enqueue_op)

      enqueue_thread = self.checkedThread(target=enqueue)
      enqueue_thread.start()

      elements_dequeued = 0
      while elements_dequeued < 250:
        # With equal probability, run Dequeue or dequeue_many.
        if random.random() > 0.5:
          self.assertEqual(elements_dequeued, dequeued_t.eval())
          elements_dequeued += 1
        else:
          count = random.randint(0, min(20, 250 - elements_dequeued))
          expected_range = np.arange(
              elements_dequeued, elements_dequeued + count, dtype=np.int32)
          self.assertAllEqual(expected_range,
                              dequeuemany_t.eval({
                                  count_placeholder: count
                              }))
          elements_dequeued += count

      q.close().run()
      enqueue_thread.join()
      self.assertEqual(0, q.size().eval())

  def testBlockingDequeueMany(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, ())
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(4)

      dequeued_elems = []

      def enqueue():
        # The enqueue_op should run after the dequeue op has blocked.
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        sess.run(enqueue_op)

      def dequeue():
        dequeued_elems.extend(sess.run(dequeued_t).tolist())

      enqueue_thread = self.checkedThread(target=enqueue)
      dequeue_thread = self.checkedThread(target=dequeue)
      enqueue_thread.start()
      dequeue_thread.start()
      enqueue_thread.join()
      dequeue_thread.join()

      self.assertAllEqual(elems, dequeued_elems)

  def testBlockingDequeueUpTo(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, ())
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_up_to(4)

      dequeued_elems = []

      def enqueue():
        # The enqueue_op should run after the dequeue op has blocked.
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        sess.run(enqueue_op)

      def dequeue():
        dequeued_elems.extend(sess.run(dequeued_t).tolist())

      enqueue_thread = self.checkedThread(target=enqueue)
      dequeue_thread = self.checkedThread(target=dequeue)
      enqueue_thread.start()
      dequeue_thread.start()
      enqueue_thread.join()
      dequeue_thread.join()

      self.assertAllEqual(elems, dequeued_elems)

  def testDequeueManyWithTensorParameter(self):
    with self.test_session():
      # Define a first queue that contains integer counts.
      dequeue_counts = [random.randint(1, 10) for _ in range(100)]
      count_q = data_flow_ops.FIFOQueue(100, dtypes_lib.int32, ())
      enqueue_counts_op = count_q.enqueue_many((dequeue_counts,))
      total_count = sum(dequeue_counts)

      # Define a second queue that contains total_count elements.
      elems = [random.randint(0, 100) for _ in range(total_count)]
      q = data_flow_ops.FIFOQueue(total_count, dtypes_lib.int32, ())
      enqueue_elems_op = q.enqueue_many((elems,))

      # Define a subgraph that first dequeues a count, then DequeuesMany
      # that number of elements.
      dequeued_t = q.dequeue_many(count_q.dequeue())

      enqueue_counts_op.run()
      enqueue_elems_op.run()

      dequeued_elems = []
      for _ in dequeue_counts:
        dequeued_elems.extend(dequeued_t.eval())
      self.assertEqual(elems, dequeued_elems)

  def testDequeueFromClosedQueue(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue()

      enqueue_op.run()
      close_op.run()
      for elem in elems:
        self.assertEqual([elem], dequeued_t.eval())

      # Expect the operation to fail due to the queue being closed.
      with self.assertRaisesRegexp(errors_impl.OutOfRangeError,
                                   "is closed and has insufficient"):
        dequeued_t.eval()

  def testBlockingDequeueFromClosedQueue(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue()

      enqueue_op.run()

      def dequeue():
        for elem in elems:
          self.assertEqual([elem], sess.run(dequeued_t))
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegexp(errors_impl.OutOfRangeError,
                                     "is closed and has insufficient"):
          sess.run(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testBlockingDequeueFromClosedEmptyQueue(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      close_op = q.close()
      dequeued_t = q.dequeue()

      def dequeue():
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegexp(errors_impl.OutOfRangeError,
                                     "is closed and has insufficient"):
          sess.run(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testBlockingDequeueManyFromClosedQueue(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, ())
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_many(4)

      enqueue_op.run()

      def dequeue():
        self.assertAllEqual(elems, sess.run(dequeued_t))
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegexp(errors_impl.OutOfRangeError,
                                     "is closed and has insufficient"):
          sess.run(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testBlockingDequeueManyButNotAllFromClosedQueue(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, ())
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_many(3)

      enqueue_op.run()

      def dequeue():
        self.assertAllEqual(elems[:3], sess.run(dequeued_t))
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegexp(errors_impl.OutOfRangeError,
                                     "is closed and has insufficient"):
          sess.run(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testDequeueUpToFromClosedQueueReturnsRemainder(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, ())
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_up_to(3)

      enqueue_op.run()

      def dequeue():
        self.assertAllEqual(elems[:3], sess.run(dequeued_t))
        self.assertAllEqual(elems[3:], sess.run(dequeued_t))

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testEnqueueManyLargerThanCapacityWithConcurrentDequeueMany(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(4, dtypes_lib.float32, ())
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_many(3)
      cleanup_dequeue_t = q.dequeue()

      def enqueue():
        sess.run(enqueue_op)

      def dequeue():
        self.assertAllEqual(elems[0:3], sess.run(dequeued_t))
        with self.assertRaises(errors_impl.OutOfRangeError):
          sess.run(dequeued_t)
        self.assertEqual(elems[3], sess.run(cleanup_dequeue_t))

      def close():
        sess.run(close_op)

      enqueue_thread = self.checkedThread(target=enqueue)
      enqueue_thread.start()

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)

      close_thread = self.checkedThread(target=close)
      close_thread.start()

      enqueue_thread.join()
      dequeue_thread.join()
      close_thread.join()

  def testClosedBlockingDequeueManyRestoresPartialBatch(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(4, (dtypes_lib.float32, dtypes_lib.float32), (
          (), ()))
      elems_a = [1.0, 2.0, 3.0]
      elems_b = [10.0, 20.0, 30.0]
      enqueue_op = q.enqueue_many((elems_a, elems_b))
      dequeued_a_t, dequeued_b_t = q.dequeue_many(4)
      cleanup_dequeue_a_t, cleanup_dequeue_b_t = q.dequeue()
      close_op = q.close()

      enqueue_op.run()

      def dequeue():
        with self.assertRaises(errors_impl.OutOfRangeError):
          sess.run([dequeued_a_t, dequeued_b_t])

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)

      close_op.run()
      dequeue_thread.join()
      # Test that the elements in the partially-dequeued batch are
      # restored in the correct order.
      for elem_a, elem_b in zip(elems_a, elems_b):
        val_a, val_b = sess.run([cleanup_dequeue_a_t, cleanup_dequeue_b_t])
        self.assertEqual(elem_a, val_a)
        self.assertEqual(elem_b, val_b)
      self.assertEqual(0, q.size().eval())

  def testBlockingDequeueManyFromClosedEmptyQueue(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, ())
      close_op = q.close()
      dequeued_t = q.dequeue_many(4)

      def dequeue():
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegexp(errors_impl.OutOfRangeError,
                                     "is closed and has insufficient"):
          sess.run(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testBlockingDequeueUpToFromClosedEmptyQueue(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, ())
      close_op = q.close()
      dequeued_t = q.dequeue_up_to(4)

      def dequeue():
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegexp(errors_impl.OutOfRangeError,
                                     "is closed and has insufficient"):
          sess.run(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testEnqueueToClosedQueue(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      enqueue_op = q.enqueue((10.0,))
      close_op = q.close()

      enqueue_op.run()
      close_op.run()

      # Expect the operation to fail due to the queue being closed.
      with self.assertRaisesRegexp(errors_impl.CancelledError, "is closed"):
        enqueue_op.run()

  def testEnqueueManyToClosedQueue(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()

      enqueue_op.run()
      close_op.run()

      # Expect the operation to fail due to the queue being closed.
      with self.assertRaisesRegexp(errors_impl.CancelledError, "is closed"):
        enqueue_op.run()

  def testBlockingEnqueueToFullQueue(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(4, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue((50.0,))
      dequeued_t = q.dequeue()

      enqueue_op.run()

      def blocking_enqueue():
        sess.run(blocking_enqueue_op)

      thread = self.checkedThread(target=blocking_enqueue)
      thread.start()
      # The dequeue ops should run after the blocking_enqueue_op has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      for elem in elems:
        self.assertEqual([elem], dequeued_t.eval())
      self.assertEqual([50.0], dequeued_t.eval())
      thread.join()

  def testBlockingEnqueueManyToFullQueue(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(4, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue_many(([50.0, 60.0],))
      dequeued_t = q.dequeue()

      enqueue_op.run()

      def blocking_enqueue():
        sess.run(blocking_enqueue_op)

      thread = self.checkedThread(target=blocking_enqueue)
      thread.start()
      # The dequeue ops should run after the blocking_enqueue_op has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      for elem in elems:
        self.assertEqual([elem], dequeued_t.eval())
        time.sleep(0.01)
      self.assertEqual([50.0], dequeued_t.eval())
      self.assertEqual([60.0], dequeued_t.eval())

      # Make sure the thread finishes before exiting.
      thread.join()

  def testBlockingEnqueueBeforeClose(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(4, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue((50.0,))
      close_op = q.close()
      dequeued_t = q.dequeue()

      enqueue_op.run()

      def blocking_enqueue():
        # Expect the operation to succeed once the dequeue op runs.
        sess.run(blocking_enqueue_op)

      enqueue_thread = self.checkedThread(target=blocking_enqueue)
      enqueue_thread.start()

      # The close_op should run after the blocking_enqueue_op has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)

      def close():
        sess.run(close_op)

      close_thread = self.checkedThread(target=close)
      close_thread.start()

      # The dequeue will unblock both threads.
      self.assertEqual(10.0, dequeued_t.eval())
      enqueue_thread.join()
      close_thread.join()

      for elem in [20.0, 30.0, 40.0, 50.0]:
        self.assertEqual(elem, dequeued_t.eval())
      self.assertEqual(0, q.size().eval())

  def testBlockingEnqueueManyBeforeClose(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(4, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue_many(([50.0, 60.0],))
      close_op = q.close()
      dequeued_t = q.dequeue()
      enqueue_op.run()

      def blocking_enqueue():
        sess.run(blocking_enqueue_op)

      enqueue_thread = self.checkedThread(target=blocking_enqueue)
      enqueue_thread.start()

      # The close_op should run after the blocking_enqueue_op has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)

      def close():
        sess.run(close_op)

      close_thread = self.checkedThread(target=close)
      close_thread.start()

      # The dequeue will unblock both threads.
      self.assertEqual(10.0, dequeued_t.eval())
      enqueue_thread.join()
      close_thread.join()
      for elem in [20.0, 30.0, 50.0, 60.0]:
        self.assertEqual(elem, dequeued_t.eval())

  def testDoesNotLoseValue(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(1, dtypes_lib.float32)
      enqueue_op = q.enqueue((10.0,))
      size_t = q.size()

      enqueue_op.run()
      for _ in range(500):
        self.assertEqual(size_t.eval(), [1])

  def testSharedQueueSameSession(self):
    with self.test_session():
      q1 = data_flow_ops.FIFOQueue(
          1, dtypes_lib.float32, shared_name="shared_queue")
      q1.enqueue((10.0,)).run()

      q2 = data_flow_ops.FIFOQueue(
          1, dtypes_lib.float32, shared_name="shared_queue")

      q1_size_t = q1.size()
      q2_size_t = q2.size()

      self.assertEqual(q1_size_t.eval(), [1])
      self.assertEqual(q2_size_t.eval(), [1])

      self.assertEqual(q2.dequeue().eval(), [10.0])

      self.assertEqual(q1_size_t.eval(), [0])
      self.assertEqual(q2_size_t.eval(), [0])

      q2.enqueue((20.0,)).run()

      self.assertEqual(q1_size_t.eval(), [1])
      self.assertEqual(q2_size_t.eval(), [1])

      self.assertEqual(q1.dequeue().eval(), [20.0])

      self.assertEqual(q1_size_t.eval(), [0])
      self.assertEqual(q2_size_t.eval(), [0])

  def testIncompatibleSharedQueueErrors(self):
    with self.test_session():
      q_a_1 = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shared_name="q_a")
      q_a_2 = data_flow_ops.FIFOQueue(15, dtypes_lib.float32, shared_name="q_a")
      q_a_1.queue_ref.op.run()
      with self.assertRaisesOpError("capacity"):
        q_a_2.queue_ref.op.run()

      q_b_1 = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shared_name="q_b")
      q_b_2 = data_flow_ops.FIFOQueue(10, dtypes_lib.int32, shared_name="q_b")
      q_b_1.queue_ref.op.run()
      with self.assertRaisesOpError("component types"):
        q_b_2.queue_ref.op.run()

      q_c_1 = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shared_name="q_c")
      q_c_2 = data_flow_ops.FIFOQueue(
          10, dtypes_lib.float32, shapes=[(1, 1, 2, 3)], shared_name="q_c")
      q_c_1.queue_ref.op.run()
      with self.assertRaisesOpError("component shapes"):
        q_c_2.queue_ref.op.run()

      q_d_1 = data_flow_ops.FIFOQueue(
          10, dtypes_lib.float32, shapes=[(1, 1, 2, 3)], shared_name="q_d")
      q_d_2 = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shared_name="q_d")
      q_d_1.queue_ref.op.run()
      with self.assertRaisesOpError("component shapes"):
        q_d_2.queue_ref.op.run()

      q_e_1 = data_flow_ops.FIFOQueue(
          10, dtypes_lib.float32, shapes=[(1, 1, 2, 3)], shared_name="q_e")
      q_e_2 = data_flow_ops.FIFOQueue(
          10, dtypes_lib.float32, shapes=[(1, 1, 2, 4)], shared_name="q_e")
      q_e_1.queue_ref.op.run()
      with self.assertRaisesOpError("component shapes"):
        q_e_2.queue_ref.op.run()

      q_f_1 = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shared_name="q_f")
      q_f_2 = data_flow_ops.FIFOQueue(
          10, (dtypes_lib.float32, dtypes_lib.int32), shared_name="q_f")
      q_f_1.queue_ref.op.run()
      with self.assertRaisesOpError("component types"):
        q_f_2.queue_ref.op.run()

  def testSelectQueue(self):
    with self.test_session():
      num_queues = 10
      qlist = list()
      for _ in xrange(num_queues):
        qlist.append(data_flow_ops.FIFOQueue(10, dtypes_lib.float32))
      # Enqueue/Dequeue into a dynamically selected queue
      for _ in xrange(20):
        index = np.random.randint(num_queues)
        q = data_flow_ops.FIFOQueue.from_list(index, qlist)
        q.enqueue((10.,)).run()
        self.assertEqual(q.dequeue().eval(), 10.0)

  def testSelectQueueOutOfRange(self):
    with self.test_session():
      q1 = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      q2 = data_flow_ops.FIFOQueue(15, dtypes_lib.float32)
      enq_q = data_flow_ops.FIFOQueue.from_list(3, [q1, q2])
      with self.assertRaisesOpError("is not in"):
        enq_q.dequeue().eval()

  def _blockingDequeue(self, sess, dequeue_op):
    with self.assertRaisesOpError("was cancelled"):
      sess.run(dequeue_op)

  def _blockingDequeueMany(self, sess, dequeue_many_op):
    with self.assertRaisesOpError("was cancelled"):
      sess.run(dequeue_many_op)

  def _blockingEnqueue(self, sess, enqueue_op):
    with self.assertRaisesOpError("was cancelled"):
      sess.run(enqueue_op)

  def _blockingEnqueueMany(self, sess, enqueue_many_op):
    with self.assertRaisesOpError("was cancelled"):
      sess.run(enqueue_many_op)

  def testResetOfBlockingOperation(self):
    with self.test_session() as sess:
      q_empty = data_flow_ops.FIFOQueue(5, dtypes_lib.float32, ())
      dequeue_op = q_empty.dequeue()
      dequeue_many_op = q_empty.dequeue_many(1)

      q_full = data_flow_ops.FIFOQueue(5, dtypes_lib.float32)
      sess.run(q_full.enqueue_many(([1.0, 2.0, 3.0, 4.0, 5.0],)))
      enqueue_op = q_full.enqueue((6.0,))
      enqueue_many_op = q_full.enqueue_many(([6.0],))

      threads = [
          self.checkedThread(
              self._blockingDequeue, args=(sess, dequeue_op)),
          self.checkedThread(
              self._blockingDequeueMany, args=(sess, dequeue_many_op)),
          self.checkedThread(
              self._blockingEnqueue, args=(sess, enqueue_op)),
          self.checkedThread(
              self._blockingEnqueueMany, args=(sess, enqueue_many_op))
      ]
      for t in threads:
        t.start()
      time.sleep(0.1)
      sess.close()  # Will cancel the blocked operations.
      for t in threads:
        t.join()

  def testBigEnqueueMany(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(5, dtypes_lib.int32, ((),))
      elem = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      enq = q.enqueue_many((elem,))
      deq = q.dequeue()
      size_op = q.size()

      enq_done = []

      def blocking_enqueue():
        enq_done.append(False)
        # This will fill the queue and then block until enough dequeues happen.
        sess.run(enq)
        enq_done.append(True)

      thread = self.checkedThread(target=blocking_enqueue)
      thread.start()

      # The enqueue should start and then block.
      results = []
      results.append(deq.eval())  # Will only complete after the enqueue starts.
      self.assertEqual(len(enq_done), 1)
      self.assertEqual(sess.run(size_op), 5)

      for _ in range(3):
        results.append(deq.eval())

      time.sleep(0.1)
      self.assertEqual(len(enq_done), 1)
      self.assertEqual(sess.run(size_op), 5)

      # This dequeue will unblock the thread.
      results.append(deq.eval())
      time.sleep(0.1)
      self.assertEqual(len(enq_done), 2)
      thread.join()

      for i in range(5):
        self.assertEqual(size_op.eval(), 5 - i)
        results.append(deq.eval())
        self.assertEqual(size_op.eval(), 5 - i - 1)

      self.assertAllEqual(elem, results)

  def testBigDequeueMany(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(2, dtypes_lib.int32, ((),))
      elem = np.arange(4, dtype=np.int32)
      enq_list = [q.enqueue((e,)) for e in elem]
      deq = q.dequeue_many(4)

      results = []

      def blocking_dequeue():
        # Will only complete after 4 enqueues complete.
        results.extend(sess.run(deq))

      thread = self.checkedThread(target=blocking_dequeue)
      thread.start()
      # The dequeue should start and then block.
      for enq in enq_list:
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        self.assertEqual(len(results), 0)
        sess.run(enq)

      # Enough enqueued to unblock the dequeue
      thread.join()
      self.assertAllEqual(elem, results)

  def testDtypes(self):
    with self.test_session() as sess:
      dtypes = [
          dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
          dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8, dtypes_lib.int64,
          dtypes_lib.uint16, dtypes_lib.bool, dtypes_lib.complex64,
          dtypes_lib.complex128
      ]
      shape = (32, 4, 128)
      q = data_flow_ops.FIFOQueue(32, dtypes, [shape[1:]] * len(dtypes))

      input_tuple = []
      for dtype in dtypes:
        np_dtype = dtype.as_numpy_dtype
        np_array = np.random.randint(-10, 10, shape)
        if dtype == dtypes_lib.bool:
          np_array = np_array > 0
        elif dtype in (dtypes_lib.complex64, dtypes_lib.complex128):
          np_array = np.sqrt(np_array.astype(np_dtype))
        else:
          np_array = np_array.astype(np_dtype)
        input_tuple.append(np_array)

      q.enqueue_many(input_tuple).run()

      output_tuple_t = q.dequeue_many(32)
      output_tuple = sess.run(output_tuple_t)

      for (input_elem, output_elem) in zip(input_tuple, output_tuple):
        self.assertAllEqual(input_elem, output_elem)

  def testDequeueEnqueueFail(self):
    with self.test_session() as session:
      q = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
      a = q.dequeue()
      b = control_flow_ops.Assert(False, ["Before enqueue"])
      with ops.control_dependencies([b]):
        c = q.enqueue(33)
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "Before enqueue" in str(e)):
        session.run([a, c])


class FIFOQueueDictTest(test.TestCase):

  def testConstructor(self):
    with ops.Graph().as_default():
      q = data_flow_ops.FIFOQueue(
          5, (dtypes_lib.int32, dtypes_lib.float32),
          names=("i", "j"),
          shared_name="foo",
          name="Q")
    self.assertTrue(isinstance(q.queue_ref, ops.Tensor))
    self.assertProtoEquals("""
      name:'Q' op:'FIFOQueueV2'
      attr { key: 'component_types' value { list {
        type: DT_INT32 type : DT_FLOAT
      } } }
      attr { key: 'shapes' value { list {} } }
      attr { key: 'capacity' value { i: 5 } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: 'foo' } }
      """, q.queue_ref.op.node_def)
    self.assertEqual(["i", "j"], q.names)

  def testConstructorWithShapes(self):
    with ops.Graph().as_default():
      q = data_flow_ops.FIFOQueue(
          5, (dtypes_lib.int32, dtypes_lib.float32),
          names=("i", "f"),
          shapes=(tensor_shape.TensorShape([1, 1, 2, 3]),
                  tensor_shape.TensorShape([5, 8])),
          name="Q")
    self.assertTrue(isinstance(q.queue_ref, ops.Tensor))
    self.assertProtoEquals("""
      name:'Q' op:'FIFOQueueV2'
      attr { key: 'component_types' value { list {
        type: DT_INT32 type : DT_FLOAT
      } } }
      attr { key: 'shapes' value { list {
        shape { dim { size: 1 }
                dim { size: 1 }
                dim { size: 2 }
                dim { size: 3 } }
        shape { dim { size: 5 }
                dim { size: 8 } }
      } } }
      attr { key: 'capacity' value { i: 5 } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: '' } }
      """, q.queue_ref.op.node_def)
    self.assertEqual(["i", "f"], q.names)

  def testEnqueueDequeueOneComponent(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(
          10, dtypes_lib.float32, shapes=((),), names="f")
      # Verify that enqueue() checks that when using names we must enqueue a
      # dictionary.
      with self.assertRaisesRegexp(ValueError, "enqueue a dictionary"):
        enqueue_op = q.enqueue(10.0)
      with self.assertRaisesRegexp(ValueError, "enqueue a dictionary"):
        enqueue_op = q.enqueue((10.0,))
      # The dictionary keys must match the queue component names.
      with self.assertRaisesRegexp(ValueError, "match names of Queue"):
        enqueue_op = q.enqueue({})
      with self.assertRaisesRegexp(ValueError, "match names of Queue"):
        enqueue_op = q.enqueue({"x": 12})
      with self.assertRaisesRegexp(ValueError, "match names of Queue"):
        enqueue_op = q.enqueue({"f": 10.0, "s": "aa"})
      enqueue_op = q.enqueue({"f": 10.0})
      enqueue_op2 = q.enqueue({"f": 20.0})
      enqueue_op3 = q.enqueue({"f": 30.0})
      # Verify that enqueue_many() checks that when using names we must enqueue
      # a dictionary.
      with self.assertRaisesRegexp(ValueError, "enqueue a dictionary"):
        enqueue_op4 = q.enqueue_many([40.0, 50.0])
      # The dictionary keys must match the queue component names.
      with self.assertRaisesRegexp(ValueError, "match names of Queue"):
        enqueue_op4 = q.enqueue_many({})
      with self.assertRaisesRegexp(ValueError, "match names of Queue"):
        enqueue_op4 = q.enqueue_many({"x": 12})
      with self.assertRaisesRegexp(ValueError, "match names of Queue"):
        enqueue_op4 = q.enqueue_many({"f": [40.0, 50.0], "s": ["aa", "bb"]})
      enqueue_op4 = q.enqueue_many({"f": [40.0, 50.0]})
      dequeue = q.dequeue()
      dequeue_2 = q.dequeue_many(2)
      sess.run(enqueue_op)
      sess.run(enqueue_op2)
      sess.run(enqueue_op3)
      sess.run(enqueue_op4)
      f = sess.run(dequeue["f"])
      self.assertEqual(10.0, f)
      f = sess.run(dequeue_2["f"])
      self.assertEqual([20.0, 30.0], list(f))
      f = sess.run(dequeue_2["f"])
      self.assertEqual([40.0, 50.0], list(f))

  def testEnqueueDequeueMultipleComponent(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(
          10, (dtypes_lib.float32, dtypes_lib.int32, dtypes_lib.string),
          shapes=((), (), ()),
          names=("f", "i", "s"))
      # Verify that enqueue() checks that when using names we must enqueue a
      # dictionary.
      with self.assertRaisesRegexp(ValueError, "enqueue a dictionary"):
        enqueue_op = q.enqueue((10.0, 123, "aa"))
      # The dictionary keys must match the queue component names.
      with self.assertRaisesRegexp(ValueError, "match names of Queue"):
        enqueue_op = q.enqueue({})
      with self.assertRaisesRegexp(ValueError, "match names of Queue"):
        enqueue_op = q.enqueue({"x": 10.0})
      with self.assertRaisesRegexp(ValueError, "match names of Queue"):
        enqueue_op = q.enqueue({"i": 12, "s": "aa"})
      with self.assertRaisesRegexp(ValueError, "match names of Queue"):
        enqueue_op = q.enqueue({"i": 123, "s": "aa", "f": 10.0, "x": 10.0})
      enqueue_op = q.enqueue({"i": 123, "s": "aa", "f": 10.0})
      enqueue_op2 = q.enqueue({"i": 124, "s": "bb", "f": 20.0})
      enqueue_op3 = q.enqueue({"i": 125, "s": "cc", "f": 30.0})
      # Verify that enqueue_many() checks that when using names we must enqueue
      # a dictionary.
      with self.assertRaisesRegexp(ValueError, "enqueue a dictionary"):
        enqueue_op4 = q.enqueue_many(([40.0, 50.0], [126, 127], ["dd", "ee"]))
      # The dictionary keys must match the queue component names.
      with self.assertRaisesRegexp(ValueError, "match names of Queue"):
        enqueue_op4 = q.enqueue_many({})
      with self.assertRaisesRegexp(ValueError, "match names of Queue"):
        enqueue_op4 = q.enqueue_many({"x": [10.0, 20.0]})
      with self.assertRaisesRegexp(ValueError, "match names of Queue"):
        enqueue_op4 = q.enqueue_many({"i": [12, 12], "s": ["aa", "bb"]})
      with self.assertRaisesRegexp(ValueError, "match names of Queue"):
        enqueue_op4 = q.enqueue_many({
            "f": [40.0, 50.0],
            "i": [126, 127],
            "s": ["dd", "ee"],
            "x": [1, 2]
        })
      enqueue_op4 = q.enqueue_many({
          "f": [40.0, 50.0],
          "i": [126, 127],
          "s": ["dd", "ee"]
      })
      dequeue = q.dequeue()
      dequeue_2 = q.dequeue_many(2)
      sess.run(enqueue_op)
      sess.run(enqueue_op2)
      sess.run(enqueue_op3)
      sess.run(enqueue_op4)
      i, f, s = sess.run([dequeue["i"], dequeue["f"], dequeue["s"]])
      self.assertEqual(123, i)
      self.assertEqual(10.0, f)
      self.assertEqual(compat.as_bytes("aa"), s)
      i, f, s = sess.run([dequeue_2["i"], dequeue_2["f"], dequeue_2["s"]])
      self.assertEqual([124, 125], list(i))
      self.assertTrue([20.0, 30.0], list(f))
      self.assertTrue([compat.as_bytes("bb"), compat.as_bytes("cc")], list(s))
      i, f, s = sess.run([dequeue_2["i"], dequeue_2["f"], dequeue_2["s"]])
      self.assertEqual([126, 127], list(i))
      self.assertTrue([40.0, 50.0], list(f))
      self.assertTrue([compat.as_bytes("dd"), compat.as_bytes("ee")], list(s))


class FIFOQueueWithTimeoutTest(test.TestCase):

  def testDequeueWithTimeout(self):
    with self.test_session(
        config=config_pb2.ConfigProto(operation_timeout_in_ms=20)) as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      self.assertEqual(
          compat.as_bytes(""), q.queue_ref.op.get_attr("container"))
      dequeued_t = q.dequeue()

      # Intentionally do not run any enqueue_ops so that dequeue will block
      # until operation_timeout_in_ms.
      with self.assertRaisesRegexp(errors_impl.DeadlineExceededError,
                                   "Timed out waiting for notification"):
        sess.run(dequeued_t)

  def testReusableAfterTimeout(self):
    with self.test_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      dequeued_t = q.dequeue()
      enqueue_op = q.enqueue(37)

      with self.assertRaisesRegexp(errors_impl.DeadlineExceededError,
                                   "Timed out waiting for notification"):
        sess.run(dequeued_t, options=config_pb2.RunOptions(timeout_in_ms=10))

      with self.assertRaisesRegexp(errors_impl.DeadlineExceededError,
                                   "Timed out waiting for notification"):
        sess.run(dequeued_t, options=config_pb2.RunOptions(timeout_in_ms=10))

      sess.run(enqueue_op)
      self.assertEqual(37, sess.run(dequeued_t))


class QueueContainerTest(test.TestCase):

  def testContainer(self):
    with ops.Graph().as_default():
      with ops.container("test"):
        q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
    self.assertEqual(
        compat.as_bytes("test"), q.queue_ref.op.get_attr("container"))


class FIFOQueueBenchmark(test.Benchmark):
  """Benchmark FIFOQueue operations."""

  def _build_graph(self):
    """Builds a graph that enqueues and dequeues a single float.

    Returns:
      A tuple with the graph init tensor and graph output tensor.
    """
    q = data_flow_ops.FIFOQueue(1, "float")
    init = q.enqueue(1.0)
    x = q.dequeue()
    q_inc = q.enqueue(x + 1)
    return init, q_inc

  # TODO(suharshs): Add benchmarks for:
  #   - different capacities of the queue
  #   - various sizes of tensors
  #   - enqueue_many, dequeue_many
  def _run(self, num_iters):
    """Benchmarks enqueueing and dequeueing from a FIFOQueue.

    Args:
      num_iters: The number of iterations to run.

    Returns:
      The duration of the run in seconds.
    """
    graph = ops.Graph()
    with graph.as_default():
      init, output = self._build_graph()
    with session_lib.Session(graph=graph) as session:
      init.run()
      _ = session.run(output)  # warm up.
      start_time = time.time()
      for _ in range(num_iters):
        _ = session.run(output)
      duration = time.time() - start_time
    print("%f secs per enqueue-dequeue" % (duration / num_iters))

    self.report_benchmark(
        name="fifo_queue", iters=num_iters, wall_time=duration / num_iters)

    return duration


if __name__ == "__main__":
  test.main()
