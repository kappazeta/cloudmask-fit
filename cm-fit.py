#!/usr/bin/python3
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

import gc
import os
import json
import pickle
import argparse
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
            "version": 1,
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
                "architecture": "unet",
                "features": ["AOT", "B01", "B02", "B03", "B04", "B05", "B06", "B08", "B8A", "B09", "B11", "B12", "WVP"],
                "pixel_window_size": 9,
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
        self.batch_size_train = 256
        self.batch_size_predict = 256
        self.num_epochs = 300
        self.features = []

        self.model = None
        self.splits = {}

    def config_from_dict(self, d):
        self.path_input_dir = d["input"]["path_dir"]
        if not os.path.isabs(self.path_input_dir):
            self.path_input_dir = os.path.abspath(self.path_input_dir)

        self.split_ratio_test = d["split"]["ratio"]["test"]
        self.split_ratio_val = d["split"]["ratio"]["val"]

        self.model_arch = d["model"]["architecture"]
        self.features = d["model"]["features"]
        self.pixel_window_size = d["model"]["pixel_window_size"]
        self.batch_size_train = d["model"]["batch_size"]
        self.batch_size_predict = d["predict"]["batch_size"]
        self.num_epochs = d["model"]["num_epochs"]

    def load_config(self, path):
        with open(path, "rt") as fi:
            self.cfg = json.load(fi)
            # TODO:: Validate config structure

            self.config_from_dict(self.cfg)

    def get_tensor_shape_x(self):
        return tf.TensorShape([
            self.batch_size_train, self.pixel_window_size, self.pixel_window_size, len(self.features)
        ])

    def get_tensor_shape_y(self):
        return tf.TensorShape([
            self.batch_size_train, len(self.classes)
        ])

    def split(self):
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
        if self.model_arch == 'a1':
            self.model = CMMA1()

        checkpoint_prefix = os.path.abspath("output/" + self.model_arch)
        self.model.set_checkpoint_prefix(checkpoint_prefix)
        self.model.set_num_epochs(self.num_epochs)
        self.model.set_batch_size(self.batch_size_train)

        self.model.construct(self.pixel_window_size, self.pixel_window_size, len(self.features), 5)
        self.model.compile()

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

        # x = dataset_train.take(1)
        # print(list(x))

        self.model.fit(dataset_train, dataset_val)

    def predict(self, path, path_weights):
        if self.model_arch == 'a1':
            self.model = CMMA1()

        self.model.set_batch_size(self.batch_size_predict)

        self.model.construct(self.pixel_window_size, self.pixel_window_size, len(self.features), 5)
        self.model.compile()

        self.model.load_weights(path_weights)

        width, height = CMBGenerator.shape(path)

        img = np.zeros((width, height), dtype=np.uint8)
        for bp, bx, by in CMBGenerator.generator_predict(
                path, self.features, self.pixel_window_size, self.batch_size_predict):
            preds = self.model.predict(bx)

            # TODO:: Store the probabilities in a 3D array in NetCDF.

            for i in range(0, len(bp)):
                p = tuple(bp[i])
                img[p] = np.argmax(preds[i])

        skio.imsave('output/prediction.png', img)


def main():
    # Parse command-line arguments.
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("-t", "--train", action="store_true", dest="train", default=True,
                   help="Train a new model.")
    p.add_argument("-p", "--predict", action="store", dest="predict", default=None,
                   help="Path to predict on.")
    p.add_argument("-w", "--weights", action="store", dest="weights", default=None,
                   help="Path to the model weights to use for prediction.")
    p.add_argument("-c", "--config", action="store", dest="path_config", default="config/config_example.json",
                   help="Path to the configuration file.")
    p.add_argument("-l", "--log", action="store", dest="logfile_path", default=None,
                   help="Path for a log file, if desired.")
    p.add_argument("-v", "--verbosity", dest="verbosity", type=int, action="store", default=3,
                   help="Level of verbosity (1 - 3).")

    args = p.parse_args()

    log = None
    try:
        # Initialize logging.
        log = ulog.init_logging(
            args.verbosity, "CloudMask Fit", "CMF",
            logfile=args.logfile_path
        )

        cmf = CMFit()
        cmf.load_config(args.path_config)

        if args.predict is not None:
            cmf.predict(args.predict, args.weights)
        elif args.train:
            cmf.split()
            cmf.train()

    except Exception as e:
        if log is not None:
            log.exception("Unhandled exception")
        else:
            print("Failed to initialize error logging")
            raise e


if __name__ == "__main__":
    main()

gc.collect()
