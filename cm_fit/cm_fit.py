# vim: set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab

# Copyright 2020 KappaZeta Ltd.
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

import os
import json
import pickle
import numpy as np
import skimage.io as skio
import tensorflow as tf

from cm_fit.util.json_codec import CMFJSONEncoder
from cm_fit.util import log as ulog
from cm_fit.model.architectures import CMMA1
from cm_fit.batch_generator import CMBGenerator


class CMFit(ulog.Loggable):
    def __init__(self):
        super(CMFit, self).__init__("CMF")

        self.cfg = {
            "version": 2,
            "input": {
                "path_dir": "input/"
            },
            "split": {
                "ratio": {
                    "test": 0.1,
                    "val": 0.1
                }
            },
            "model": {
                "architecture": "a1",
                "features": ["AOT", "B01", "B02", "B03", "B04", "B05", "B06", "B08", "B8A", "B09", "B11", "B12", "WVP"],
                "pixel_window_size": 9,
            },
            "train": {
                "learning_rate": 1E-4,
                "batch_size": 256,
                "num_epochs": 10
            },
            "predict": {
                "batch_size": 256
            }
        }

        self.classes = [
            "UNDEFINED", "CLEAR", "CLOUD_SHADOW", "SEMI_TRANSPARENT_CLOUD", "CLOUD"
        ]

        self.split_ratio_test = 0.1
        self.split_ratio_val = 0.1

        self.model_arch = ""
        self.path_input_dir = ""
        self.pixel_window_size = 9
        self.features = []

        self.learning_rate = 1E-4
        self.batch_size_train = 256
        self.batch_size_predict = 256
        self.num_epochs = 300

        self.model = None
        self.splits = {}

    def config_from_dict(self, d):
        """
        Load configuration from a dictionary.
        :param d: Dictionary with the configuration tree.
        """
        self.path_input_dir = d["input"]["path_dir"]
        if not os.path.isabs(self.path_input_dir):
            self.path_input_dir = os.path.abspath(self.path_input_dir)

        self.split_ratio_test = d["split"]["ratio"]["test"]
        self.split_ratio_val = d["split"]["ratio"]["val"]

        self.model_arch = d["model"]["architecture"]
        self.features = d["model"]["features"]
        self.pixel_window_size = d["model"]["pixel_window_size"]

        self.learning_rate = d["train"]["learning_rate"]
        self.batch_size_train = d["train"]["batch_size"]
        self.batch_size_predict = d["predict"]["batch_size"]
        self.num_epochs = d["train"]["num_epochs"]

    def load_config(self, path):
        """
        Load configuration from a JSON file.
        :param path: Path to the JSON file.
        """
        with open(path, "rt") as fi:
            self.cfg = json.load(fi)
            # TODO:: Validate config structure

            self.config_from_dict(self.cfg)

    def get_tensor_shape_x(self):
        """
        Get the shape of the X (input) tensor.
        :return: An instance of tensorflow.TensorShape.
        """
        return tf.TensorShape([
            self.batch_size_train, self.pixel_window_size, self.pixel_window_size, len(self.features)
        ])

    def get_tensor_shape_y(self):
        """
        Get the shape of the Y (output) tensor.
        :return: An instance of tensorflow.TensorShape.
        """
        return tf.TensorShape([
            self.batch_size_train, len(self.classes)
        ])

    def split(self):
        """
        Read the files in self.path_input_dir, and split them according to self.split_ratio_val, self.split_ratio_test.
        The results are written to output/splits.json.
        """
        self.splits = CMBGenerator.split(
            self.path_input_dir,
            ratio_val=self.split_ratio_val,
            ratio_test=self.split_ratio_test
        )

        self.log.info(
            (
                "Dataset split as follows:\n"
                " Total: {total}\n"
                " Train: {train} ({train_p:.2f}%)\n"
                " Validation: {val} ({val_p:.2f}%)\n"
                " Test: {test} ({test_p:.2f}%)"
            ).format(
                total=self.splits['total'],
                train=len(self.splits['train']),
                val=len(self.splits['val']),
                test=len(self.splits['test']),
                train_p=100 * len(self.splits['train']) / self.splits['total'],
                val_p=100 * len(self.splits['val']) / self.splits['total'],
                test_p=100 * len(self.splits['test']) / self.splits['total']
            )
        )

        path_splits = os.path.abspath("output/splits.json")
        with open(path_splits, "wt") as fo:
            json.dump(self.splits, fo, cls=CMFJSONEncoder, indent=4)

    def train(self):
        """
        Fit a model to the training dataset (obtained from a splitting operation).
        """
        if self.model_arch == 'a1':
            self.model = CMMA1()

        # Propagate configuration parameters.
        checkpoint_prefix = os.path.abspath("output/" + self.model_arch)
        self.model.set_checkpoint_prefix(checkpoint_prefix)
        self.model.set_num_epochs(self.num_epochs)
        self.model.set_batch_size(self.batch_size_train)
        self.model.set_learning_rate(self.learning_rate)

        # Construct and compile the model.
        self.model.construct(self.pixel_window_size, self.pixel_window_size, len(self.features), 5)
        self.model.compile()

        # Initialize the dataset for training.
        args = [
            pickle.dumps(self.splits), 'train',
            self.features, self.pixel_window_size, self.batch_size_train, self.num_epochs, len(self.classes)
        ]
        dataset_train = tf.data.Dataset.from_generator(
            CMBGenerator.generator_train,
            args=args,
            output_types=(tf.float32, tf.float32),
            output_shapes=(self.get_tensor_shape_x(), self.get_tensor_shape_y())
        )

        # Initialize the dataset for validation.
        args = [
            pickle.dumps(self.splits), 'val',
            self.features, self.pixel_window_size, self.batch_size_train, self.num_epochs, len(self.classes)
        ]
        dataset_val = tf.data.Dataset.from_generator(
            CMBGenerator.generator_train,
            args=args,
            output_types=(tf.float32, tf.float32),
            output_shapes=(self.get_tensor_shape_x(), self.get_tensor_shape_y())
        )

        self.model.set_num_samples(len(self.splits['train']), len(self.splits['val']))

        # Fit the model, storing weights in output/.
        self.model.fit(dataset_train, dataset_val)

    def predict(self, path, path_weights):
        """
        Predict on a data cube, using model weights from a specific file.
        Prediction results are stored in output/prediction.png.
        :param path: Path to the input data cube.
        :param path_weights: Path to the model weights.
        """
        if self.model_arch == 'a1':
            self.model = CMMA1()

        # Propagate configuration parameters.
        self.model.set_batch_size(self.batch_size_predict)

        # Construct and compile the model.
        self.model.construct(self.pixel_window_size, self.pixel_window_size, len(self.features), 5)
        self.model.compile()

        # Load model weights.
        self.model.load_weights(path_weights)

        # Create an array for storing the segmentation mask.
        width, height = CMBGenerator.shape(path)
        img = np.zeros((width, height), dtype=np.uint8)

        # Iterate over the dataset for prediction.
        for bp, bx, by in CMBGenerator.generator_predict(
                path, self.features, self.pixel_window_size, self.batch_size_predict):
            preds = self.model.predict(bx)

            # TODO:: Store the probabilities in a 3D array in NetCDF.

            # Update the segmentation mask.
            # TODO:: This is slow. Optimize it.
            for i in range(0, len(bp)):
                p = tuple(bp[i])
                img[p] = np.argmax(preds[i])

        # Save the segmentation mask.
        skio.imsave('output/prediction.png', img)

