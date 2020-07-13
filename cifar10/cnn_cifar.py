#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

N_IMAGES = 50000
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def plot_cifar10(xtrain, ytrain):
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(xtrain[i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(CLASS_NAMES[ytrain[i][0]])
    plt.show()


def create_small_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def create_vgg19():
    vgg19 = tf.keras.applications.VGG19(include_top=False,
                                        weights=None,
                                        input_shape=(32, 32, 3),
                                        classes=10)
    vgg19.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return vgg19


def predict(model, xtest):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(xtest[i], cmap=plt.cm.binary)

        label = model.predict(xtest[i])

        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(CLASS_NAMES[label[0]])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--verbose', type=int, choices=[0, 1], default=0)
    parser.add_argument('--steps_per_epoch', type=int)

    args = parser.parse_args()

    steps_per_epoch = args.steps_per_epoch
    if not steps_per_epoch:
        steps_per_epoch = N_IMAGES // args.batchsize


    (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.cifar10.load_data()
    xtrain = xtrain / 255.0
    xtest = xtest / 255.0

    model = create_small_model()

    history = model.fit(xtrain, ytrain,
                        epochs=args.epochs,
                        verbose=args.verbose,
                        batch_size=args.batchsize,
                        steps_per_epoch=steps_per_epoch)
