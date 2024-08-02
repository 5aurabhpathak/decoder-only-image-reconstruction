"""
MIT license notice
© 2024 Saurabh Pathak. All rights reserved
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Purpose: Dataloader classes and utilities for loading them
"""
import functools
import typing

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit

import hparams


def get_data(config, training=True):
    """
    reads in dataset defined in config and returns training set or test set conditioned on whether training==True
    :param config: config dict
    :param training: True for training set, False for testing set
    :return: training set or test set conditioned on whether training==True
    """
    dataset_name = config.dataset
    if dataset_name == 'cifar100':
        dataset = tf.keras.datasets.cifar100
    elif dataset_name == 'cifar10':
        dataset = tf.keras.datasets.cifar10
    elif dataset_name == 'mnist':
        dataset = tf.keras.datasets.mnist
    elif dataset_name == 'fmnist':
        dataset = tf.keras.datasets.fashion_mnist
    else:
        raise ValueError(f'unknown dataset: {dataset_name}')

    try:
        (x_train, y_train), (x_test, y_test) = dataset.load_data()
    except AttributeError:
        raise ValueError(f'{dataset.__name__} must have load_data() defined')

    x, y = (x_train, y_train) if training else (x_test, y_test)

    # select a fraction of data if 0 < config.examples_face < 1
    if config.examples_frac:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=config.examples_frac, random_state=hparams.RANDOM_SEED)
        _, inds = next(sss.split(x, y))
        x, y = x[inds], y[inds]

    if 'mnist' in dataset_name:
        x = x[..., np.newaxis]

    x = x.astype(np.float32)
    return np.arange(x.shape[0]), x, y


def scaler(i, x, config):
    """
    scales x part of data to specified range in config
    :param i: index, passes through unchanged
    :param x: data x
    :param config: config dict
    :return: scaled data
    """
    scale = config.scale_data

    if scale == 'standard':
        x = (x - 127.5) / 127.5
    elif scale == 'zero_one':
        x = x / 255.
    elif isinstance(scale, typing.Callable):
        x = scale(x)
    elif scale != 'identity':
        raise ValueError(f'Unknown scaler: {str(scale)}')

    return i, x


def get_data_dims(config):
    """
    get the data dimensions
    :param config: config dict
    :return: number of samples in train and val, shape of output dimensions
    """
    #TODO: need to define a simpler logic. Currently this requires reading the whole dataset
    _, x, _ = get_data(config, training=True)
    _, val_x, _ = get_data(config, training=False)
    return x.shape[0], val_x.shape[0], x[0].ravel().shape[0]


def augmentations(config):
    """
    creates model that applies data augmentation
    :param config: config dict
    :return: keras model that performs data augmentation
    """
    data_augmentation = dict(brightness=tf.keras.layers.RandomBrightness(hparams.augmentations.brightness,
                                                                         value_range=(-1, 1.)),
                             contrast=tf.keras.layers.RandomRotation(hparams.augmentations.contrast,
                                                                     fill_mode='constant'),
                             zoom=tf.keras.layers.RandomZoom(hparams.augmentations.height_factor,
                                                             hparams.augmentations.width_factor, fill_mode='constant'),
                             translate=tf.keras.layers.RandomTranslation(hparams.augmentations.height_factor,
                                                                         hparams.augmentations.width_factor,
                                                                         fill_mode='constant'))

    data_augmentation = tf.keras.Sequential([v for k, v in data_augmentation.items()
                                             if hparams.augmentations.enabled[k] and
                                             not hparams.augmentations.disable_all])
    data_augmentation.add(tf.keras.layers.Lambda(lambda x:
                                                 tf.clip_by_value(x, -1. if config.scale_data == 'standard' else 0.,
                                                                  1.)))
    data_augmentation.add(tf.keras.layers.Flatten())

    data_augmentation.compile(run_eagerly=False)
    return data_augmentation


def get_dataset(config, training=True):
    """
    creates tf.data pipeline
    :param config: config dict
    :param training: training flag to return training or test dataset
    :return: tf.data pipeline
    """
    i, x, _ = get_data(config, training=training)
    ds = tf.data.Dataset.from_tensor_slices((i, x))

    if config.dataset != 'caltech256':
        scaler_fn = functools.partial(scaler, config=config)
        ds = ds.map(scaler_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache().shuffle(4096, reshuffle_each_iteration=True).batch(config.batchsize,
                                                                       drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    data_augmentation = augmentations(config)

    def augment_batch(i_data, x_data):
        """
        adds data augmentations on input images
        :param i_data: data index
        :param x_data: data x
        :return: augmented data x; i_data is passed through unchanged
        """
        x_data = data_augmentation(x_data, training=training)
        return i_data, x_data

    ds = ds.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)
