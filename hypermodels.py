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

Purpose: Hypermodels for keras tuner hyperparameter tuning
"""
import keras_tuner
import tensorflow as tf

import dataloader
import hparams
import train


class HyperModel(keras_tuner.HyperModel):
    """
    hypermodel class. This class can build models by varying following hyperparameters:
    learning rate: min_value=1e-4, max_value=.1, step=10, sampling='log'
    This hparam can be prefixed as well. In that case, it is not tuned.
    """

    def __init__(self, config, *args, **kwargs):
        """
        init
        :param config: config dict
        :param args: superclass args
        :param kwargs: superclass kwargs
        """
        super().__init__(*args, **kwargs)
        self.config = config

    def build(self, hp):
        """
        build the model using hp
        :param hp: keras tuner hyperparameters instance
        :return: keras model
        """
        self.config.lr = hp.Choice('lr', hparams.LR)

        model = train.Model(self.config, name='model')
        model.compile(run_eagerly=False,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.lr),
                      val_optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.lr),
                      metrics='mae')
        return model

    def fit(self, hp, model, *args, **kwargs):
        """
        fit the model with keras tuner
        :param hp: keras tuner hyperparameters instance
        :param model: keras model
        :param args: passed on to model.fit() args
        :param kwargs: passed on to model.fit() kwargs
        :return: keras model.fit() result
        """
        # set tensorboard attribute in model instance. needed when visualizing histograms
        if self.config.histogram_freq or self.config.visualize_after_epochs:
            tboard_handle = [x for x in kwargs['callbacks'] if isinstance(x, tf.keras.callbacks.TensorBoard)]
            if len(tboard_handle):
                model.tboard = tboard_handle[0]

        dataset = dataloader.get_dataset(self.config, training=True)
        validation_dataset = dataloader.get_dataset(self.config, training=False)
        kwargs['callbacks'].append(train.VisualizationCallback())

        return super().fit(hp, model, dataset, *args, validation_data=validation_dataset, **kwargs)
