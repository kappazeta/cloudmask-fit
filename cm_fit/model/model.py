# vim: set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab

# Copyright 2020 KappaZeta Ltd.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from cm_fit.util import log


class CMModel(log.Loggable):
    """
    A generic model class to be subclassed by specific model architecture classes.
    """

    def __init__(self, log_abbrev='CMM'):
        super(CMModel, self).__init__(log_abbrev)

        self.input_shape = (512, 512, 10)
        self.output_shape = (10,)
        self.model = None

        self.learning_rate = 1E-4
        self.num_train_samples = 0
        self.num_val_samples = 0
        self.batch_size = 0
        self.num_epochs = 300
        
        self.monitored_metric = 'categorical_accuracy'

        self.path_checkpoint = ''

    def construct(self, width, height, num_channels, num_categories):
        """
        Just an abstract placeholder function to be overloaded by subclasses.
        :param width: Width of a single sample (must be an odd number).
        :param height: Height of a single sample (must be an odd number).
        :param num_channels: Number of features used.
        :param num_categories: Number of output classes.
        :return:
        """
        raise NotImplementedError()

    def compile(self):
        """
        Compile the model for the Adam optimizer.
        :return:
        """
        with tf.name_scope('Optimizer'):
            l_op = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.model.compile(optimizer=l_op, loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.summary()

        return self.model

    def load_weights(self, path):
        """
        Load model weights from a file.
        :param path: Path to the model weights file.
        """
        self.model.load_weights(path)

    def set_learning_rate(self, lr):
        """
        Set initial learning rate for model fitting.
        :param lr: Learning rate, usually 1E-4 or less. It is recommended to adjust in orders of magnitude.
        """
        self.learning_rate = lr

    def set_checkpoint_prefix(self, prefix):
        """
        Set a path prefix for model training.
        :param prefix: Path prefix, should end with a filename prefix.
        """
        self.path_checkpoint = prefix + '_{epoch:03d}-{val_accuracy:.2f}.hdf5'

    def set_num_samples(self, num_train_samples, num_val_samples):
        """
        Set number of samples in the train and val datasets.
        :param num_train_samples: Number of samples for fitting.
        :param num_val_samples: Number of samples for validation.
        """
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples

    def set_batch_size(self, batch_size):
        """
        Set the number of samples per batch.
        :param batch_size: Number of samples per batch.
        """
        self.batch_size = batch_size

    def set_num_epochs(self, num_epochs):
        """
        Set the number of epochs (full iterations with all batches) to fit for.
        :param num_epochs: Number of epochs.
        """
        self.num_epochs = num_epochs

    def fit(self, dataset_train, dataset_val):
        """
        Train the model, producing model weights files as specified by the checkpoint path.
        :param dataset_train: Tensorflow Dataset to use for training.
        :param dataset_val: Tensorflow Dataset to use for validation.
        """
        callbacks = []

        with tf.name_scope('Callbacks'):
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor="accuracy", patience=30)
            callbacks.append(early_stopping)

            if self.path_checkpoint != '':
                model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    self.path_checkpoint, monitor=self.monitored_metric,
                    save_weights_only=True, mode='auto'
                )
                callbacks.append(model_checkpoint)

            lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="accuracy", factor=0.5, patience=10, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0
            )
            callbacks.append(lr_reducer)

        num_train_batches_per_epoch = self.num_train_samples / self.batch_size
        num_val_batches_per_epoch = self.num_val_samples / self.batch_size

        # TODO:: Duplicate a random number of samples, to fill batches.

        with tf.name_scope('Training'):
            self.model.fit(
                x=dataset_train, validation_data=dataset_val, callbacks=callbacks, epochs=self.num_epochs,
                steps_per_epoch=num_train_batches_per_epoch, validation_steps=num_val_batches_per_epoch
            )

    def predict(self, dataset_pred):
        """
        Predict on a dataset.
        :param dataset_pred: Dataset (numpy ndarray or Tensorflow Dataset) to predict on.
        :return: Numpy array of class probabilities [[p_class1, p_class2, ...], [p_class1, p_class2, ...]].
        """
        preds = self.model.predict(dataset_pred, self.batch_size)

        return preds
