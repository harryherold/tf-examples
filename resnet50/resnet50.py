import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import resnet50
from skimage.transform import resize


def imagenet_generator(dataset, batch_size=32, num_classes=1000):
    images = np.empty((batch_size, 224, 224, 3))
    labels = np.empty((batch_size))
    while True:
        count = 0
        for sample in tfds.as_numpy(dataset):
            image = sample["image"]
            label = sample["label"]
            x = resize(image, (224, 224), anti_aliasing=True)
            x = tf.keras.preprocessing.image.img_to_array(x)
            x = resnet50.preprocess_input(x)

            images[count%batch_size] = x
            labels[count%batch_size] = label
            count += 1
            if (count%batch_size == 0):
                yield images, labels


def start_yes_or_no():
    while True:
        answer = input("Start training? [Y/N] : ").upper()
        if answer == 'Y' or answer == 'N':
            return True if answer == 'Y' else False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tf_record_path')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--images', type=int, default=1281167)
    parser.add_argument('--steps_per_epoch', type=int)
    parser.add_argument('--input_type', type=str, choices=['dataset', 'generator'], default='dataset')
    parser.add_argument('--verbose', type=int, choices=[0, 1], default=0)
    parser.add_argument('--yes', action='store_true')

    args = parser.parse_args()

    dataset, info = tfds.load('imagenet2012',
                            download=False,
                            with_info=True,
                            split='train',
                            data_dir=args.tf_record_path)

    classes = info.features["label"].num_classes
    shape = info.features['image'].shape
    width = 224
    height = 224
    channels = 3
    shape = (width, height, channels)
    batch_size = args.batchsize
    epochs = args.epochs
    images = args.images
    start = True

    if not args.steps_per_epoch:
        steps_per_epoch = images // batch_size
    else:
        steps_per_epoch = args.steps_per_epoch

    print('\n##########################')
    print('# Training configuration #')
    print('##########################')
    print('Epochs:          {}'.format(epochs))
    print('Images:          {}'.format(images))
    print('Batch size:      {}'.format(batch_size))
    print('Steps per epoch: {}'.format(steps_per_epoch))
    print('Input type:      {}\n'.format(args.input_type))

    if not args.yes:
        start = start_yes_or_no()

    if not start:
        exit(0)

    model = resnet50.ResNet50(weights=None, input_shape=shape, classes=classes)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'],
                optimizer=tf.keras.optimizers.Adam())

    if args.input_type == 'dataset':
        dataset = dataset.map(lambda x: (tf.image.resize(x['image'], (width, height)), x['label']),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE).take(images).batch(batch_size)

        model.fit(dataset,
                 steps_per_epoch=steps_per_epoch,
                 epochs=epochs,
                 verbose=args.verbose)
    else:
        model.fit(imagenet_generator(dataset.take(images),
                                     batch_size=batch_size,
                                     num_classes=classes),
                  steps_per_epoch=steps_per_epoch,
                  epochs=epochs,
                  verbose=args.verbose)
