#!/usr/bin/env python3
import argparse
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
import time


class MeasureTimeCallback(keras.callbacks.Callback):

    def __init__(self):
        self.times = []
        self.t_start = 0.0
        self.t_batch = 0.0

    def on_train_end(self, logs=None):
        self.times.append(self.t_batch)
        self.t_batch = 0.0
        self.t_start = 0.0

    def on_batch_begin(self, batch, logs=None):
        self.t_start = time.time()

    def on_batch_end(self, batch, logs=None):
        t_end = time.time()
        self.t_batch += t_end - self.t_start

def load_input_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # reshape data to fit model
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)

    # one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32,
                                  kernel_size=(3, 3),
                                  activation='relu',
                                  input_shape=(28, 28, 1)))

    model.add(keras.layers.Conv2D(64,
                                  kernel_size=(3, 3),
                                  activation='relu'))

    model.add(keras.layers.Conv2D(10,
                                  kernel_size=(24, 24),
                                  activation='softmax'))

    model.add(keras.layers.Flatten())

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--steps_per_epoch', type=int, default=1)
    parser.add_argument('--device', type=str, default='CPU:0')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--convert-to-onnx', action='store_true')

    args = parser.parse_args()

    if args.summary:
        model.summary()
        exit(0)

    if args.convert_to_onnx:
        import keras2onnx
        onnx_model = keras2onnx.convert_keras(model, name='mnist_cnn')
        keras2onnx.save_model(onnx_model, 'model.onnx')
        model.save('keras_model')
        exit(0)

    batchsize = args.batchsize
    steps = args.steps_per_epoch
    epochs = args.epochs
    repeat = args.repeat

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    (x_train, y_train, x_test, y_test) = load_input_data()

    try:
        with tf.device("/{}".format(args.device)):

            # Warm up
            model.fit(x_train,
                      y_train,
                      steps_per_epoch=steps,
                      batch_size=batchsize,
                      epochs=1,
                      verbose=0)

            for _ in range(repeat):
                model.fit(x_train,
                          y_train,
                          steps_per_epoch=steps,
                          batch_size=batchsize,
                          epochs=epochs,
                          verbose=0,
                          callbacks=[callback])

        print("{},".format(callback.times))

    except RuntimeError as e:
        print(e)
