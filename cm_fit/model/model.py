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
import numpy as np

from cm_fit.util import log
from cm_fit.model.unet_original import Unet
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from tensorflow.keras.callbacks import TensorBoard
import time


class CMModel(log.Loggable):
    """
    A generic model class to be subclassed by specific model architecture classes.
    """

    def __init__(self, log_abbrev='CMM'):
        super(CMModel, self).__init__(log_abbrev)

        self.input_shape = (512, 512, 10)
        self.output_shape = (10,)
        self.model = Unet(input_size=(512, 512, 13))

        self.learning_rate = 0.001
        self.num_train_samples = 0
        self.num_val_samples = 0
        self.batch_size = 0
        self.num_epochs = 3
        self.class_weights = [1,1,5.7,3.6,1.3,1]

        # Accuracy, precision, recall, f1, iou
        self.METRICS_SET = {"accuracy": tf.keras.metrics.Accuracy(), "categorical_acc": tf.keras.metrics.CategoricalAccuracy(),
                            "recall": tf.keras.metrics.Recall(), "precision": tf.keras.metrics.Precision(),
                            "iou": tf.keras.metrics.MeanIoU(num_classes=6), 'f1': self.custom_f1}
        self.monitored_metric = self.METRICS_SET["iou"]

        self.path_checkpoint = ''

    def construct(self, width, height, num_channels, num_categories,
                  layers=False, units=False, pretrained_weights=False):
        """
        Just an abstract placeholder function to be overloaded by subclasses.
        :param width: Width of a single sample (must be an odd number).
        :param height: Height of a single sample (must be an odd number).
        :param num_channels: Number of features used.
        :param num_categories: Number of output classes.
        :return:
        """
        raise NotImplementedError()

    def compile(self, loss_name):
        """
        Compile the model for the Adam optimizer.
        :return:
        """
        with tf.name_scope('Optimizer'):
            l_op = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        if loss_name == "dice_loss":
            self.model.compile(optimizer=l_op, loss=self.dice_loss, #sample_weight_mode="temporal",
                               metrics=[self.METRICS_SET["precision"], self.METRICS_SET["recall"],
                                        self.METRICS_SET["categorical_acc"], self.METRICS_SET['f1'],
                                        self.METRICS_SET['iou']])
        elif loss_name == "categorical_crossentropy":
            self.model.compile(optimizer=l_op, loss='categorical_crossentropy', #sample_weight_mode="temporal",
                               metrics=[self.METRICS_SET["precision"], self.METRICS_SET["recall"],
                                        self.METRICS_SET["categorical_acc"], self.METRICS_SET['f1'],
                                        self.METRICS_SET['iou']])
        elif loss_name == "cat_dice_loss":
            self.model.compile(optimizer=l_op, loss='categorical_crossentropy', #sample_weight_mode="temporal",
                               metrics=[self.METRICS_SET["precision"], self.METRICS_SET["recall"],
                                        self.METRICS_SET["categorical_acc"], self.METRICS_SET['f1'],
                                        self.METRICS_SET['iou']])
        elif loss_name == "weighted_loss":
            self.model.compile(optimizer=l_op, loss=self.weighted_dice_loss, #sample_weight_mode="temporal",
                               metrics=[self.METRICS_SET["precision"], self.METRICS_SET["recall"],
                                        self.METRICS_SET["categorical_acc"], self.METRICS_SET['f1'],
                                        self.METRICS_SET['iou']])
        else:
            self.model.compile(optimizer=l_op, loss='categorical_crossentropy', #sample_weight_mode="temporal",
                               metrics=[self.METRICS_SET["precision"], self.METRICS_SET["recall"],
                                        self.METRICS_SET["categorical_acc"], self.METRICS_SET['f1'],
                                        self.METRICS_SET['iou']])
        print("new optimizer")
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
        self.path_checkpoint = prefix + '_{epoch:03d}-{val_loss:.2f}.hdf5'

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

    @staticmethod
    def custom_f1(y_true, y_pred):
        def recall_m(y_true, y_pred):
            TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

            recall = TP / (Positives + K.epsilon())
            return recall

        def precision_m(y_true, y_pred):
            TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

            precision = TP / (Pred_Positives + K.epsilon())
            return precision

        precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

        f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
        weighted_f1 = f1 * K.round(K.clip(y_true, 0, 1)) / K.sum(K.round(K.clip(y_true, 0, 1)))
        weighted_f1 = K.sum(weighted_f1)

        return f1

    @staticmethod
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives + K.epsilon())
        return precision

    @staticmethod
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives + K.epsilon())
        return recall

    @staticmethod
    def accuracy_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        TN = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
        FP = K.sum(K.round(K.clip((1-y_true) * y_pred, 0, 1)))
        FN = K.sum(K.round(K.clip(y_true * (1-y_pred), 0, 1)))

        accuracy = (TP + TN) / (TP + TN + FP + FN + K.epsilon())
        return accuracy

    @staticmethod
    def dice_loss(y_true, y_pred):
        def dice_coef(y_true, y_pred, smooth=1):
            """
            Dice = (2*|X & Y|)/ (|X|+ |Y|)
                 =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
            ref: https://arxiv.org/pdf/1606.04797v1.pdf
            """
            intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
            return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)
        return 1 - dice_coef(y_true, y_pred)

    @staticmethod
    def weighted_dice_loss(y_true, y_pred):
        def recall_m(y_true, y_pred):
            TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

            recall = TP / (Positives + K.epsilon())
            return recall

        def precision_m(y_true, y_pred):
            TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

            precision = TP / (Pred_Positives + K.epsilon())
            return precision

        precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

        f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
        weights = K.constant([1, 1, 4.6, 2.8, 1.2, 1])
        weighted_loss = f1 * K.sum(y_true * weights, axis=-1)
        return weighted_loss

    @staticmethod
    def bce_dice_loss(y_true, y_pred):
        def dice_coef(y_true, y_pred, smooth=1):
            """
            Dice = (2*|X & Y|)/ (|X|+ |Y|)
                 =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
            ref: https://arxiv.org/pdf/1606.04797v1.pdf
            """
            intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
            return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)
        return 0.5 * K.categorical_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

    @staticmethod
    def cat_dice_loss(y_true, y_pred):
        def weighted_categorical_crossentropy(y_true, y_pred):
            # weights = K.variable([0.5,2.0,0.0])
            weights = K.constant([1,1,5.7,3.6,1.3,1])

            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)

            return loss

        def dice_loss(y_true, y_pred):
            def dice_coef(y_true, y_pred, smooth=1):
                """
                Dice = (2*|X & Y|)/ (|X|+ |Y|)
                     =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
                ref: https://arxiv.org/pdf/1606.04797v1.pdf
                """
                intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
                return (2. * intersection + smooth) / (
                            K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

            return 1 - dice_coef(y_true, y_pred)

        return weighted_categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    @staticmethod
    def get_confusion_matrix(y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred)
        cm_multi = multilabel_confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5])
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_multi_norm = cm_multi.astype('float') / cm_multi.sum(axis=2)[:, :, np.newaxis]
        return cm, cm_normalized, cm_multi, cm_multi_norm

    def fit(self, dataset_train, dataset_val, model_name):
        """
        Train the model, producing model weights files as specified by the checkpoint path.
        :param dataset_train: Tensorflow Dataset to use for training.
        :param dataset_val: Tensorflow Dataset to use for validation.
        """
        callbacks = []

        with tf.name_scope('Callbacks'):
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_custom_f1", mode='max', patience=40)
            callbacks.append(early_stopping)

            if self.path_checkpoint != '':
                model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    self.path_checkpoint, monitor="val_custom_f1",
                    save_weights_only=True, mode='max'
                )
                callbacks.append(model_checkpoint)

            lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_custom_f1", factor=0.5, patience=20, mode='max', min_delta=0.0001, cooldown=0, min_lr=0
            )
            callbacks.append(lr_reducer)
            tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name), histogram_freq=1, profile_batch=2)
            callbacks.append(tensorboard)

        num_train_batches_per_epoch = self.num_train_samples // self.batch_size
        num_val_batches_per_epoch = self.num_val_samples // self.batch_size

        # TODO:: Duplicate a random number of samples, to fill batches.

        with tf.name_scope('Training'):
            history = self.model.fit_generator(
                dataset_train, validation_data=dataset_val, callbacks=callbacks, epochs=self.num_epochs,
                steps_per_epoch=num_train_batches_per_epoch, validation_steps=num_val_batches_per_epoch
            )
        return history

    def predict(self, dataset_pred):
        """
        Predict on a dataset.
        :param dataset_pred: Dataset (numpy ndarray or Tensorflow Dataset) to predict on.
        :return: Numpy array of class probabilities [[p_class1, p_class2, ...], [p_class1, p_class2, ...]].
        """
        preds = self.model.predict_generator(dataset_pred)

        return preds

    def predict_classes_gen(self, dataset_pred):
        """
        Predict on a dataset.
        :param dataset_pred: Dataset (numpy ndarray or Tensorflow Dataset) to predict on.
        :return: Numpy array of class probabilities [[p_class1, p_class2, ...], [p_class1, p_class2, ...]].
        """
        preds = self.model.predict_generator(dataset_pred)

        preds = np.argmax(preds, axis=3)

        return preds
