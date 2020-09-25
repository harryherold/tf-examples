#!/usr/bin/env python3
import argparse
import json
import tensorflow as tf

import tensorflow_datasets as tfds
import time

def print_model_summary():
    classes = 1000
    width = 224
    height = 224
    channels = 3
    shape = (width, height, channels)

    vgg16 = tf.keras.applications.VGG16(input_shape=shape,
                                        weights=None,
                                        classes=classes)
    vgg16.compile(optimizer='sgd',
                    loss='sparse_categorical_crossentropy')
    vgg16.summary()


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--steps_per_epoch', type=int)
parser.add_argument('--imagenet_path', type=str)
parser.add_argument('--summary', action='store_true')

args = parser.parse_args()

if args.summary:
    print_model_summary()
    exit(0)

dataset, info = tfds.load('imagenet2012',
                          download=False,
                          with_info=True,
                          split='train',
                          data_dir=args.imagenet_path)

classes = info.features["label"].num_classes

width = 224
height = 224
channels = 3
shape = (width, height, channels)
batchsize = args.batchsize
steps = args.steps_per_epoch
images = steps * batchsize
epochs = args.epochs
repeat = args.repeat

try:
    with tf.device('/GPU:0'):
        vgg16 = tf.keras.applications.VGG16(input_shape=shape,
                                            weights=None,
                                            classes=classes)
        vgg16.compile(optimizer='sgd',
                      loss='sparse_categorical_crossentropy')

        dataset = dataset.map(lambda x: (tf.image.resize(x['image'], (width, height)),
                                         x['label']),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Warm up
        i = iter(tfds.as_numpy(dataset.batch(batchsize)))
        image, label = next(i)
        vgg16.fit(image, label, epochs=1, verbose=0)

        times = []
        data = dataset.take(steps * batchsize).batch(batchsize).repeat()
        for _ in range(repeat):
            s = time.time()
            vgg16.fit(data, epochs=epochs, steps_per_epoch=steps, verbose=0)
            e = time.time()
            times.append(e - s)
        print("{},".format(times))

except RuntimeError as e:
    print(e)
