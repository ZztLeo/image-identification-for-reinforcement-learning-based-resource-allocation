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
"""Tests for the experimental input pipeline ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path
import shutil
import tempfile

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class ListFilesDatasetOpTest(test.TestCase):

  def setUp(self):
    self.tmp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.tmp_dir, ignore_errors=True)

  def _touchTempFiles(self, filenames):
    for filename in filenames:
      open(path.join(self.tmp_dir, filename), 'a').close()

  def testEmptyDirectory(self):
    dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'))
    with self.test_session() as sess:
      itr = dataset.make_one_shot_iterator()
      next_element = itr.get_next()
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testSimpleDirectory(self):
    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'))
    with self.test_session() as sess:
      itr = dataset.make_one_shot_iterator()
      next_element = itr.get_next()

      full_filenames = []
      produced_filenames = []
      for filename in filenames:
        full_filenames.append(
            compat.as_bytes(path.join(self.tmp_dir, filename)))
        produced_filenames.append(compat.as_bytes(sess.run(next_element)))
      self.assertItemsEqual(full_filenames, produced_filenames)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(itr.get_next())

  def testEmptyDirectoryInitializer(self):
    filename_placeholder = array_ops.placeholder(dtypes.string, shape=[])
    dataset = dataset_ops.Dataset.list_files(filename_placeholder)

    with self.test_session() as sess:
      itr = dataset.make_initializable_iterator()
      next_element = itr.get_next()
      sess.run(
          itr.initializer,
          feed_dict={filename_placeholder: path.join(self.tmp_dir, '*')})

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testSimpleDirectoryInitializer(self):
    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    filename_placeholder = array_ops.placeholder(dtypes.string, shape=[])
    dataset = dataset_ops.Dataset.list_files(filename_placeholder)

    with self.test_session() as sess:
      itr = dataset.make_initializable_iterator()
      next_element = itr.get_next()
      sess.run(
          itr.initializer,
          feed_dict={filename_placeholder: path.join(self.tmp_dir, '*')})

      full_filenames = []
      produced_filenames = []
      for filename in filenames:
        full_filenames.append(
            compat.as_bytes(path.join(self.tmp_dir, filename)))
        produced_filenames.append(compat.as_bytes(sess.run(next_element)))

      self.assertItemsEqual(full_filenames, produced_filenames)

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(itr.get_next())

  def testFileSuffixes(self):
    filenames = ['a.txt', 'b.py', 'c.py', 'd.pyc']
    self._touchTempFiles(filenames)

    filename_placeholder = array_ops.placeholder(dtypes.string, shape=[])
    dataset = dataset_ops.Dataset.list_files(filename_placeholder)

    with self.test_session() as sess:
      itr = dataset.make_initializable_iterator()
      next_element = itr.get_next()
      sess.run(
          itr.initializer,
          feed_dict={filename_placeholder: path.join(self.tmp_dir, '*.py')})

      full_filenames = []
      produced_filenames = []
      for filename in filenames[1:-1]:
        full_filenames.append(
            compat.as_bytes(path.join(self.tmp_dir, filename)))
        produced_filenames.append(compat.as_bytes(sess.run(next_element)))
      self.assertItemsEqual(full_filenames, produced_filenames)

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(itr.get_next())

  def testFileMiddles(self):
    filenames = ['a.txt', 'b.py', 'c.pyc']
    self._touchTempFiles(filenames)

    filename_placeholder = array_ops.placeholder(dtypes.string, shape=[])
    dataset = dataset_ops.Dataset.list_files(filename_placeholder)

    with self.test_session() as sess:
      itr = dataset.make_initializable_iterator()
      next_element = itr.get_next()
      sess.run(
          itr.initializer,
          feed_dict={filename_placeholder: path.join(self.tmp_dir, '*.py*')})

      full_filenames = []
      produced_filenames = []
      for filename in filenames[1:]:
        full_filenames.append(
            compat.as_bytes(path.join(self.tmp_dir, filename)))
        produced_filenames.append(compat.as_bytes(sess.run(next_element)))

      self.assertItemsEqual(full_filenames, produced_filenames)

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(itr.get_next())

  def testNoShuffle(self):
    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    # Repeat the list twice and ensure that the order is the same each time.
    # NOTE(mrry): This depends on an implementation detail of `list_files()`,
    # which is that the list of files is captured when the iterator is
    # initialized. Otherwise, or if e.g. the iterator were initialized more than
    # once, it's possible that the non-determinism of `tf.matching_files()`
    # would cause this test to fail. However, it serves as a useful confirmation
    # that the `shuffle=False` argument is working as intended.
    # TODO(b/73959787): Provide some ordering guarantees so that this test is
    # more meaningful.
    dataset = dataset_ops.Dataset.list_files(
        path.join(self.tmp_dir, '*'), shuffle=False).repeat(2)
    with self.test_session() as sess:
      itr = dataset.make_one_shot_iterator()
      next_element = itr.get_next()

      full_filenames = []
      produced_filenames = []
      for filename in filenames * 2:
        full_filenames.append(
            compat.as_bytes(path.join(self.tmp_dir, filename)))
        produced_filenames.append(compat.as_bytes(sess.run(next_element)))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(itr.get_next())
      self.assertItemsEqual(full_filenames, produced_filenames)
      self.assertEqual(produced_filenames[:len(filenames)],
                       produced_filenames[len(filenames):])


if __name__ == '__main__':
  test.main()
