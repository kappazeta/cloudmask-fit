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
from cm_fit.model.unet_original import Unet
from cm_fit.data_loader.data_generator import DataGenerator
from cm_fit.data_loader.utils import generate_splits
from cm_fit.training.utils import set_normalization
from tensorflow.keras.utils import Sequence
from keras.callbacks import ModelCheckpoint
from shutil import copyfile


class CMInit(ulog.Loggable):
    def __init__(self):
        super(CMInit, self).__init__("CMF")

        self.cfg = {
            "version": 2,
            "input": {
                "path_dir": "input/"
            },
            "split": {
                "ratio": {
                    "test": 0.1,
                    "val": 0.2
                }
            },
            "model": {
                "architecture": "a1",
                "features": ["AOT", "B01", "B02", "B03", "B04", "B05", "B06", "B08", "B8A", "B09", "B11", "B12", "WVP"],
                "pixel_window_size": 9,
            },
            "train": {
                "learning_rate": 1E-4,
                "batch_size": 16,
                "num_epochs": 10
            },
            "predict": {
                "batch_size": 16
            }
        }

        self.classes = [
            "UNDEFINED", "CLEAR", "CLOUD_SHADOW", "SEMI_TRANSPARENT_CLOUD", "CLOUD"
        ]

        self.split_ratio_test = 0.1
        self.split_ratio_val = 0.2

        self.model_arch = ""
        self.path_data_dir = ""
        self.experiment_name = ""
        self.experiment_res_folder = "results"
        self.prediction_path = self.experiment_res_folder + "/prediction"
        self.validation_path = self.experiment_res_folder + "/validation"
        self.meta_data_path = self.experiment_res_folder + "/meta_data"
        self.pixel_window_size = 9
        self.features = []

        self.learning_rate = 1E-4
        self.batch_size_train = 16
        self.batch_size_predict = 16
        self.num_epochs = 10
        self.dim = (512, 512)
        self.params = {'path_input': self.path_data_dir,
                       'batch_size': self.batch_size_train,
                       'features': self.features,
                       'dim': self.dim,
                       'num_classes': len(self.classes),
                       'shuffle': True}

        self.model = None
        self.splits = {}
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def config_from_dict(self, d):
        """
        Load configuration from a dictionary.
        :param d: Dictionary with the configuration tree.
        """
        self.path_data_dir = d["input"]["data_dir"]
        self.experiment_name = d["experiment_name"]
        self.experiment_res_folder = "results/" + self.experiment_name
        self.prediction_path = self.experiment_res_folder + "/prediction"
        self.validation_path = self.experiment_res_folder + "/validation"
        self.meta_data_path = self.experiment_res_folder + "/meta_data"
        self.checkpoints_path = self.experiment_res_folder + "/checkpoints"
        if not os.path.isabs(self.path_data_dir):
            self.path_data_dir = os.path.abspath(self.path_data_dir)

        self.split_ratio_test = d["split"]["ratio"]["test"]
        self.split_ratio_val = d["split"]["ratio"]["val"]

        self.model_arch = d["model"]["architecture"]
        self.features = d["model"]["features"]
        self.pixel_window_size = d["model"]["pixel_window_size"]

        self.learning_rate = d["train"]["learning_rate"]
        self.batch_size_train = d["train"]["batch_size"]
        self.batch_size_predict = d["predict"]["batch_size"]
        self.num_epochs = d["train"]["num_epochs"]
        self.create_folders()

    def create_folders(self):
        if not os.path.exists('results'):
            os.mkdir('results')
        if not os.path.exists(self.experiment_res_folder):
            os.mkdir(self.experiment_res_folder)
        if not os.path.exists(self.prediction_path):
            os.mkdir(self.prediction_path)
        if not os.path.exists(self.validation_path):
            os.mkdir(self.validation_path)
        if not os.path.exists(self.meta_data_path):
            os.mkdir(self.meta_data_path)
        if not os.path.exists(self.checkpoints_path):
            os.mkdir(self.checkpoints_path)

    def load_config(self, path):
        """
        Load configuration from a JSON file.
        :param path: Path to the JSON file.
        """
        with open(path, "rt") as fi:
            self.cfg = json.load(fi)
            # TODO:: Validate config structure

            self.config_from_dict(self.cfg)
            copyfile(path, self.meta_data_path+"/config.json")

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

    def save_to_img_contrast(self, path, data):
        """
        Save a 2D array of uint8 values into an image.
        :param path: Path to the image file.
        :param data: The array to save.
        """
        data *= 51
        skio.imsave(path, data)

    def split(self):
        """
        Read the files in self.path_input_dir, and split them according to self.split_ratio_val, self.split_ratio_test.
        The results are written to output/splits.json.
        """
        self.splits = generate_splits(self.path_data_dir, self.split_ratio_val, self.split_ratio_test)

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

        path_splits = os.path.abspath(self.meta_data_path + "/splits.json")
        with open(path_splits, "wt") as fo:
            json.dump(self.splits, fo, cls=CMFJSONEncoder, indent=4)

    def get_model_by_name(self, name):
        if self.model_arch in ARCH_MAP:
            self.model = ARCH_MAP[name]()
            return self.model
        else:
            raise ValueError(("Unsupported architecture \"{}\"."
                              " Only the following architectures are supported: {}.").format(name, ARCH_MAP.keys()))

    def save_masks_contrast(self, path_image, prediction, classification, saving_path):
        path_image = path_image.rstrip()
        filename_image = path_image.split('/')[-1].split(".")[0]
        for i, label in enumerate(self.classes):
            saving_filename = saving_path + filename_image + label
            current_class = prediction[:, :, i]
            #current_class[current_class >= 0.5] = 255
            #current_class[current_class < 0.5] = 0
            skio.imsave(saving_filename+".png", current_class)
        classification *= 51
        skio.imsave(saving_path + "/" + filename_image + ".png", classification)
        print(filename_image)
        return True

    def train(self):
        """
        Fit a model to the training dataset (obtained from a splitting operation).
        """
        self.params["features"] = self.features
        self.params["batch_size"] = self.batch_size_train

        training_generator = DataGenerator(self.splits['train'], **self.params)
        validation_generator = DataGenerator(self.splits['val'], **self.params)
        self.get_model_by_name(self.model_arch)
        # Propagate configuration parameters.
        checkpoint_prefix = os.path.abspath(self.checkpoints_path + "/unet_init_")
        self.model.set_checkpoint_prefix(checkpoint_prefix)
        self.model.set_num_epochs(self.num_epochs)
        self.model.set_batch_size(self.batch_size_train)
        self.model.set_learning_rate(self.learning_rate)

        # Construct and compile the model.
        self.model.construct(self.dim[0], self.dim[1], len(self.features), len(self.classes))
        self.model.model.summary()
        self.model.compile()

        self.model.set_num_samples(len(self.splits['train']), len(self.splits['val']))

        train_std, train_means = set_normalization(training_generator, self.splits['train'], 5)
        val_std, val_means = set_normalization(validation_generator, self.splits['val'], 1)

        # Fit the model, storing weights in checkpoints/.
        self.model.fit(training_generator,
                       validation_generator)

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
        self.model.construct(self.dim[0], self.dim[1], len(self.features), len(self.classes))
        self.model.compile()

        # Load model weights.
        self.model.load_weights(path_weights)

        # Create an array for storing the segmentation mask.
        probabilities = np.zeros((self.dim[0], self.dim[1], len(self.classes)), dtype=np.float)
        class_mask = np.zeros((self.dim[0], self.dim[1]), dtype=np.uint8)
        self.params["features"] = self.features
        self.params["batch_size"] = self.batch_size_predict
        self.params["shuffle"] = False

        # Read splits again
        path_splits = os.path.abspath(self.meta_data_path+"/splits.json")
        with open(path_splits, "r") as fo:
            dictionary = json.load(fo)

        valid_generator = DataGenerator(dictionary['val'], **self.params)

        predictions = self.model.predict(valid_generator)

        classes = self.model.predict_classes_gen(valid_generator)

        print(predictions.shape)
        for i, prediction in enumerate(predictions):
            self.save_masks_contrast(dictionary['val'][i], prediction, classes[i], self.prediction_path)

        """self.save_to_nc("output/model_v1/prediction.nc", "probabilities", probabilities)
        self.save_to_img("output/model_v1/prediction.png", class_mask)
        self.save_to_img_contrast("output/model_v1/prediction_contrast.png", class_mask)"""

    def validation(self, path_original, path_predicted):
        """
        Validate predicted data
        :param path_original: Path to the input data cube.
        :param path_predicted: Path to the model weights.
        """

        # Read splits again
        path_splits = os.path.abspath("output/model_v1/splits.json")
        with open(path_splits, "r") as fo:
            dictionary = json.load(fo)

        valid_generator = DataGenerator(dictionary['val'], **self.params)

        predictions = self.model.predict(valid_generator)

        classes = self.model.predict_classes_gen(valid_generator)

        print(predictions.shape)
        for i, prediction in enumerate(predictions):
            self.save_masks_contrast(dictionary['val'][i], prediction, classes[i], "output/model_v1/")

        """self.save_to_nc("output/model_v1/prediction.nc", "probabilities", probabilities)
        self.save_to_img("output/model_v1/prediction.png", class_mask)
        self.save_to_img_contrast("output/model_v1/prediction_contrast.png", class_mask)"""