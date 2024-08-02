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

Purpose: decoder only generator
"""
import io

import tensorflow as tf
from matplotlib import pyplot as plt

import dataloader
import sequential


class Model(tf.keras.Model):
    """represents decoder only image generator"""

    def __init__(self, config, *args, **kwargs):
        """
        init
        :param config: configuration dictionary
        :param args: additional arguments for superclass
        :param kwargs: additional keyword arguments for superclass
        """
        super().__init__(*args, **kwargs)
        self.config = config

        init_func = tf.zeros
        self.latent_space = tf.Variable(init_func((config.n_samples, config.latent_dims)), trainable=True)
        self.val_latent_space = tf.Variable(init_func((config.val_n_samples, config.latent_dims)), trainable=True)

        self.model = sequential.get_model(config)
        self.val_optimizer = None

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        if 'gradients' in config.trackers:
            self.gradient_trackers = [tf.keras.metrics.Mean(name='latent_gradients')] + [
                tf.keras.metrics.Mean(name=f'{lyr.name}_gradients') for lyr in self.model.layers
                if isinstance(lyr, tf.keras.layers.Dense)]

    def build(self, input_shape):
        """
        build the model and print the internal model summary
        :param input_shape: shape of the input tensor
        """
        super().build(input_shape)
        self.model.summary()

    def compile(self, **kwargs):
        """
        compiles the internal model, also sets val_optimizer for training the latent space on validation data
        :param kwargs: compile kwargs
        """
        self.val_optimizer = kwargs.pop('val_optimizer')
        self.val_optimizer.build([self.val_latent_space])
        super().compile(weighted_metrics=[], **kwargs)

    def call(self, inputs, training=False):
        """
        wrapper around the internal model, reshapes the latent dims appropriately if the internal model is convolutional
        :param inputs: input tensor
        :param training: training flag
        :return: internal model output
        """
        return self.model(inputs, training=training)

    @tf.function
    def _helper(self, data, *, training=True):
        """
        helper function for training and validation on a single batch
        :param data: input data
        :param training: if set also updates internal model weights, else only updates latent space using a separate
        optimizer
        :return: training/validation metrics
        """
        vector_indices, images = data
        if training:
            latent_means = self.latent_space
            optimizer = self.optimizer
        else:
            latent_means = self.val_latent_space
            optimizer = self.val_optimizer

        with tf.GradientTape() as tape:
            vectors = tf.gather(latent_means, vector_indices)
            generated_images = self(vectors, training=training)
            total_loss = tf.reduce_mean(tf.losses.mse(generated_images, images))

        trainable_vars = [latent_means]
        if training:
            trainable_vars.extend(self.model.trainable_variables)

        gradients = tape.gradient(total_loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(images, generated_images)

        if 'gradients' in self.config.trackers:
            i = 0
            for g in gradients:
                if len(g.shape) == 1:
                    continue
                self.gradient_trackers[i].update_state(g)
                i += 1

        self.loss_tracker.update_state(total_loss)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def train_step(self, data):
        """
        training step on a single batch
        :param data: input data
        :return: training metrics
        """
        return self._helper(data, training=True)

    @tf.function
    def test_step(self, data):
        """
        test step on a single batch
        :param data: input data
        :return: validation metrics
        """
        return self._helper(data, training=False)


class VisualizationCallback(tf.keras.callbacks.Callback):
    """Callback to visualize the train and validation latent spaces periodically during training"""

    def __init__(self):
        """init"""
        super().__init__()
        self.config = None
        self.trn_file_writer = None
        self.tst_file_writer = None
        self.trn_labels = None
        self.val_labels = None
        self.image_shape = None

    def set_model(self, model):
        """
        sets the model under study
        :param model: model
        """
        self.model = model
        self.config = model.config
        self.trn_file_writer = model.tboard._train_writer
        self.tst_file_writer = model.tboard._val_writer

        val_data = dataloader.get_data(self.config, training=False)
        self.image_shape = val_data[1][0].shape

        if self.config.latent_dims > 2:
            print('VisalizationCallback: Can not visualize more than two dimensions. Will not plot.')

        trn_data = dataloader.get_data(self.config, training=True)
        self.trn_labels = trn_data[2]
        self.val_labels = val_data[2]

    def on_train_begin(self, logs=None):
        """
        visualize the latent spaces before training begins
        :param logs: logs (unused)
        """
        self._helper(epoch=0)

    @staticmethod
    def finish_plot():
        """
        finalizes a pyplot
        :return: a converted tensor of the plot to a png image to be visualized on tensorboard later
        """
        plt.grid(axis='both')
        plt.gca().set_axisbelow(True)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return tf.image.decode_png(buf.getvalue(), channels=4)[tf.newaxis]

    def plot(self):
        """
        plots the latent spaces for training and validation data
        :return: png images corresponding to respective plots to be visualized on tensorboard later
        """

        def plot_helper(latent_space, labels, cmap, **kwargs):
            """
            helper function to plot a selected latent space
            :param latent_space: numpy array describing the latent space
            :param labels: labels corresponding to classes in the data for color coding
            :param cmap: colormap to be used by the pyplot
            :param kwargs: other optional kwargs to pass on to plt.scatter()
            """
            x, y = latent_space[:, 0], latent_space[:, 1]
            plt.scatter(x, y, c=labels, cmap=cmap, s=1, **kwargs)

        trn_space = self.model.latent_space.numpy()
        val_space = self.model.val_latent_space.numpy()

        plt.figure(figsize=(5, 5))
        plot_helper(trn_space, labels=self.trn_labels, cmap='jet')
        trn_image = self.finish_plot()

        plt.figure(figsize=(5, 5))
        plot_helper(trn_space, labels='gray', cmap=None, alpha=.3)
        plot_helper(val_space, labels=self.val_labels, cmap='jet')
        val_image = self.finish_plot()

        return trn_image, val_image

    def on_epoch_end(self, epoch, logs=None):
        """
        plot the latent spaces corresponding to the training and validation data and generate sample images
        every n steps, where n == self.config.visualize_after_epochs
        :param epoch: current epoch counter
        :param logs: incoming logs (unused)
        """
        epoch += 1
        if self.config.visualize_after_epochs and not epoch % self.config.visualize_after_epochs:
            self._helper(epoch=epoch)

    def _helper(self, epoch=None):
        """
        helper function to visualize the latent spaces corresponding to the training and validation data
        :param epoch: current epoch counter (incremented by 1, since default count starts at 0 otherwise)
        """
        if self.config.latent_dims == 2:
            trn_image, val_image = self.plot()
        else:
            trn_image, val_image = None, None

        with self.trn_file_writer.as_default():
            tf.summary.scalar('lmean', tf.reduce_mean(self.model.latent_space), step=epoch)
            tf.summary.scalar('lvar', tf.math.reduce_variance(self.model.latent_space), step=epoch)

            if trn_image is not None:
                tf.summary.image('latent_space', trn_image, step=epoch)

        with self.tst_file_writer.as_default():
            tf.summary.scalar('lmean', tf.reduce_mean(self.model.val_latent_space), step=epoch)
            tf.summary.scalar('lvar', tf.math.reduce_variance(self.model.val_latent_space), step=epoch)

            if val_image is not None:
                tf.summary.image('latent_space', val_image, step=epoch)
