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
"""Integration tests for Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras._impl import keras
from tensorflow.python.keras._impl.keras import testing_utils
from tensorflow.python.layers import core as tf_core_layers
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class KerasIntegrationTest(test.TestCase):

  def test_vector_classification_declarative(self):
    with self.test_session():
      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=200,
          test_samples=100,
          input_shape=(10,),
          num_classes=2)
      y_train = keras.utils.to_categorical(y_train)
      y_test = keras.utils.to_categorical(y_test)

      model = keras.models.Sequential([
          keras.layers.Dense(16,
                             activation='relu',
                             input_shape=x_train.shape[1:]),
          keras.layers.Dropout(0.1),
          keras.layers.Dense(y_train.shape[-1], activation='softmax')
      ])
      model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])
      history = model.fit(x_train, y_train, epochs=10, batch_size=16,
                          validation_data=(x_test, y_test),
                          verbose=2)
      self.assertGreater(history.history['val_acc'][-1], 0.85)

  def test_vector_classification_functional(self):
    with self.test_session():
      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=200,
          test_samples=100,
          input_shape=(10,),
          num_classes=2)
      y_train = keras.utils.to_categorical(y_train)
      y_test = keras.utils.to_categorical(y_test)

      inputs = keras.layers.Input(shape=x_train.shape[1:])
      x = keras.layers.Dense(16, activation='relu')(inputs)
      x = keras.layers.Dropout(0.1)(x)
      outputs = keras.layers.Dense(y_train.shape[-1], activation='softmax')(x)

      model = keras.models.Model(inputs, outputs)
      model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])
      history = model.fit(x_train, y_train, epochs=10, batch_size=16,
                          validation_data=(x_test, y_test),
                          verbose=2)
      self.assertGreater(history.history['val_acc'][-1], 0.85)

  def test_temporal_classification_declarative(self):
    with self.test_session():
      np.random.seed(1336)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=200,
          test_samples=100,
          input_shape=(4, 8),
          num_classes=2)
      y_train = keras.utils.to_categorical(y_train)
      y_test = keras.utils.to_categorical(y_test)

      model = keras.models.Sequential()
      model.add(keras.layers.LSTM(5, return_sequences=True,
                                  input_shape=x_train.shape[1:]))
      model.add(keras.layers.GRU(y_train.shape[-1], activation='softmax'))
      model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
      history = model.fit(x_train, y_train, epochs=10, batch_size=16,
                          validation_data=(x_test, y_test),
                          verbose=2)
      self.assertGreater(history.history['val_acc'][-1], 0.80)

  def test_image_classification_declarative(self):
    with self.test_session():
      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=200,
          test_samples=100,
          input_shape=(8, 8, 3),
          num_classes=2)
      y_train = keras.utils.to_categorical(y_train)
      y_test = keras.utils.to_categorical(y_test)

      model = keras.models.Sequential()
      model.add(keras.layers.Conv2D(
          8, 3,
          activation='relu',
          input_shape=x_train.shape[1:]))
      model.add(keras.layers.BatchNormalization())
      model.add(keras.layers.Conv2D(
          8, 3,
          padding='same',
          activation='relu'))
      model.add(keras.layers.GlobalMaxPooling2D())
      model.add(keras.layers.Dense(y_train.shape[-1], activation='softmax'))
      model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
      history = model.fit(x_train, y_train, epochs=10, batch_size=16,
                          validation_data=(x_test, y_test),
                          verbose=2)
      self.assertGreater(history.history['val_acc'][-1], 0.85)

  def test_video_classification_functional(self):
    with self.test_session():
      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=200,
          test_samples=100,
          input_shape=(4, 8, 8, 3),
          num_classes=3)
      y_train = keras.utils.to_categorical(y_train)
      y_test = keras.utils.to_categorical(y_test)

      inputs = keras.layers.Input(shape=x_train.shape[1:])
      x = keras.layers.TimeDistributed(
          keras.layers.Conv2D(4, 3, activation='relu'))(inputs)
      x = keras.layers.BatchNormalization()(x)
      x = keras.layers.TimeDistributed(keras.layers.GlobalMaxPooling2D())(x)
      x = keras.layers.Conv1D(8, 3, activation='relu')(x)
      x = keras.layers.Flatten()(x)
      outputs = keras.layers.Dense(y_train.shape[-1], activation='softmax')(x)

      model = keras.models.Model(inputs, outputs)
      model.compile(loss='categorical_crossentropy',
                    optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.8),
                    metrics=['accuracy'])
      history = model.fit(x_train, y_train, epochs=10, batch_size=16,
                          validation_data=(x_test, y_test),
                          verbose=2)
      self.assertGreater(history.history['val_acc'][-1], 0.70)

  def test_vector_classification_shared_sequential(self):
    # Test that Sequential models that feature internal updates
    # and internal losses can be shared.
    with self.test_session():
      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=200,
          test_samples=100,
          input_shape=(10,),
          num_classes=2)
      y_train = keras.utils.to_categorical(y_train)
      y_test = keras.utils.to_categorical(y_test)

      base_model = keras.models.Sequential([
          keras.layers.Dense(16,
                             activation='relu',
                             kernel_regularizer=keras.regularizers.l2(1e-5),
                             bias_regularizer=keras.regularizers.l2(1e-5),
                             input_shape=x_train.shape[1:]),
          keras.layers.BatchNormalization(),
      ])
      x = keras.layers.Input(x_train.shape[1:])
      y = base_model(x)
      y = keras.layers.Dense(y_train.shape[-1], activation='softmax')(y)
      model = keras.models.Model(x, y)
      model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])
      self.assertEqual(len(model.losses), 2)
      self.assertEqual(len(model.updates), 2)
      history = model.fit(x_train, y_train, epochs=10, batch_size=16,
                          validation_data=(x_test, y_test),
                          verbose=2)
      self.assertGreater(history.history['val_acc'][-1], 0.84)

  def test_vector_classification_shared_model(self):
    # Test that functional models that feature internal updates
    # and internal losses can be shared.
    with self.test_session():
      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=200,
          test_samples=100,
          input_shape=(10,),
          num_classes=2)
      y_train = keras.utils.to_categorical(y_train)
      y_test = keras.utils.to_categorical(y_test)

      inputs = keras.layers.Input(x_train.shape[1:])
      x = keras.layers.Dense(16,
                             activation='relu',
                             kernel_regularizer=keras.regularizers.l2(1e-5),
                             bias_regularizer=keras.regularizers.l2(1e-5),
                             input_shape=x_train.shape[1:])(inputs)
      x = keras.layers.BatchNormalization()(x)
      base_model = keras.models.Model(inputs, x)

      x = keras.layers.Input(x_train.shape[1:])
      y = base_model(x)
      y = keras.layers.Dense(y_train.shape[-1], activation='softmax')(y)
      model = keras.models.Model(x, y)
      model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])
      history = model.fit(x_train, y_train, epochs=10, batch_size=16,
                          validation_data=(x_test, y_test),
                          verbose=2)
      self.assertGreater(history.history['val_acc'][-1], 0.85)

  def test_embedding_with_clipnorm(self):
    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Embedding(input_dim=1, output_dim=1))
      model.compile(optimizer=keras.optimizers.SGD(clipnorm=0.1), loss='mse')
      model.fit(np.array([[0]]), np.array([[[0.5]]]), epochs=1)

  def test_using_tf_layers_in_keras_sequential_model(self):
    with self.test_session():
      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=200,
          test_samples=100,
          input_shape=(10,),
          num_classes=2)

      model = keras.models.Sequential()
      model.add(tf_core_layers.Dense(32, activation=nn.relu, input_shape=(10,)))
      model.add(tf_core_layers.Dense(2, activation=nn.softmax))
      model.summary()

      y_train = keras.utils.to_categorical(y_train)
      y_test = keras.utils.to_categorical(y_test)
      model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
      history = model.fit(x_train, y_train, epochs=10, batch_size=16,
                          validation_data=(x_test, y_test),
                          verbose=0)
      self.assertGreater(history.history['val_acc'][-1], 0.85)

  def test_using_tf_layers_in_keras_functional_model(self):
    with self.test_session():
      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=200,
          test_samples=100,
          input_shape=(10,),
          num_classes=2)
      y_train = keras.utils.to_categorical(y_train)
      y_test = keras.utils.to_categorical(y_test)

      inputs = keras.Input(shape=(10,))
      x = tf_core_layers.Dense(32, activation=nn.relu)(inputs)
      outputs = tf_core_layers.Dense(2, activation=nn.softmax)(x)
      model = keras.Model(inputs, outputs)
      model.summary()

      model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
      history = model.fit(x_train, y_train, epochs=10, batch_size=16,
                          validation_data=(x_test, y_test),
                          verbose=0)
      self.assertGreater(history.history['val_acc'][-1], 0.85)


if __name__ == '__main__':
  test.main()
