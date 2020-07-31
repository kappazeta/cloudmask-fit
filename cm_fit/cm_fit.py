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
import netCDF4 as nc
import skimage.io as skio
import tensorflow as tf

from cm_fit.util.json_codec import CMFJSONEncoder
from cm_fit.util import log as ulog
from cm_fit.model.architectures import ARCH_MAP
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

    def save_to_nc(self, path, name, data):
        """
        Set a variable in a NetCDF file.
        :param path: Path to the NetCDF file.
        :param name: Name of the variable.
        :param data: Data to set.
        """
        with nc.Dataset(path, 'w') as root:
            dimensions = [
                ("x", data.shape[0]),
                ("y", data.shape[1]),
                ("c", data.shape[2])
            ]
            dimension_names = [d[0] for d in dimensions]

            for key, val in dimensions:
                if key not in root.dimensions.keys():
                    root.createDimension(key, val)

            if name not in root.variables.keys():
                variable = root.createVariable(name, "f4", dimensions=dimension_names, zlib=True, complevel=9, endian="little")
            else:
                variable = root[name]
            variable[:, :, :] = data

    def save_to_img(self, path, data):
        """
        Save a 2D array of uint8 values into an image.
        :param path: Path to the image file.
        :param data: The array to save.
        """
        skio.imsave(path, data)

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

    def get_model_by_name(self, name):
        if self.model_arch in ARCH_MAP:
            self.model = ARCH_MAP[name]()
            return self.model
        else:
            raise ValueError(("Unsupported architecture \"{}\"."
                              " Only the following architectures are supported: {}.").format(name, ARCH_MAP.keys()))

    def train(self):
        """
        Fit a model to the training dataset (obtained from a splitting operation).
        """
        self.get_model_by_name(self.model_arch)

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
        self.get_model_by_name(self.model_arch)

        # Propagate configuration parameters.
        self.model.set_batch_size(self.batch_size_predict)

        # Construct and compile the model.
        self.model.construct(self.pixel_window_size, self.pixel_window_size, len(self.features), 5)
        self.model.compile()

        # Load model weights.
        self.model.load_weights(path_weights)

        # Create an array for storing the segmentation mask.
        width, height = CMBGenerator.shape(path)
        probabilities = np.zeros((width, height, len(self.classes)), dtype=np.float)
        class_mask = np.zeros((width, height), dtype=np.uint8)

        # Iterate over the dataset for prediction.
        for bp, bx, by in CMBGenerator.generator_predict(
                path, self.features, self.pixel_window_size, self.batch_size_predict):
            preds = self.model.predict(bx)

            # Update the segmentation mask.
            # TODO:: This is slow. Optimize it.
            for i in range(0, len(bp)):
                probabilities[bp[i][0], bp[i][1], :] = preds[i]
                class_mask[bp[i][0], bp[i][1]] = np.argmax(preds[i])

        # Save the segmentation mask.
        self.save_to_nc("output/prediction.nc", "probabilities", probabilities)
        self.save_to_img("output/prediction.png", class_mask)
