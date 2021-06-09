import os
import json
import pickle
import numpy as np
import netCDF4 as nc
import skimage.io as skio
import tensorflow as tf
from PIL import Image
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cm_fit.util.json_codec import CMFJSONEncoder
from cm_fit.util import log as ulog
from cm_fit.model.architectures import ARCH_MAP
from cm_fit.data_loader.data_generator import DataGenerator
from cm_fit.data_loader.data_utils import generate_splits
from cm_fit.training.train_utils import set_normalization
from cm_fit.plot.train_history import draw_history_plots
from tensorflow.keras.utils import Sequence
from keras.callbacks import ModelCheckpoint
from shutil import copyfile
from cm_fit.plot.train_history import plot_confusion_matrix, draw_4lines
from sklearn.metrics import confusion_matrix


class CMFit(ulog.Loggable):
    def __init__(self, png_mode=False):
        super(CMFit, self).__init__("CMF")

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
                "batch_size": 6,
                "num_epochs": 10
            },
            "predict": {
                "batch_size": 1
            }
        }

        self.classes = [
            "UNDEFINED", "CLEAR", "CLOUD_SHADOW", "SEMI_TRANSPARENT_CLOUD", "CLOUD", "MISSING"
        ]

        self.split_ratio_test = 0.1
        self.split_ratio_val = 0.2

        self.model_arch = ""
        self.path_data_dir = ""
        self.experiment_name = ""
        self.experiment_res_folder = "results"
        self.prediction_path = self.experiment_res_folder + "/prediction"
        self.validation_path = self.experiment_res_folder + "/validation"
        self.test_path = self.experiment_res_folder + "/test"
        self.meta_data_path = self.experiment_res_folder + "/meta_data"
        self.checkpoints_path = self.experiment_res_folder + "/checkpoints"
        self.plots_path = self.experiment_res_folder + "/plots"
        self.dataset_comparison_path = self.experiment_res_folder + "/cmix_comparison"
        self.pixel_window_size = 9
        self.features = []
        self.label_set = "Label"
        self.normalization = "std"
        self.product_level = "L2A"

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
        self.loss_name = "dice_loss"
        self.splits = {}
        if png_mode:
            self.png_iterator = True
        else:
            self.png_iterator = False
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        self.log.info("Detected physical devices: {}".format(physical_devices))
        if len(physical_devices) > 0:
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
        self.test_path = self.experiment_res_folder + "/test"
        self.meta_data_path = self.experiment_res_folder + "/meta_data"
        self.checkpoints_path = self.experiment_res_folder + "/checkpoints"
        self.plots_path = self.experiment_res_folder + "/plots"
        self.dataset_comparison_path = self.experiment_res_folder + "/cmix_comparison"
        if not os.path.isabs(self.path_data_dir):
            self.path_data_dir = os.path.abspath(self.path_data_dir)

        if "product_level" in d["input"]:
            self.product_level = d["input"]["product_level"]

        self.split_ratio_test = d["split"]["ratio"]["test"]
        self.split_ratio_val = d["split"]["ratio"]["val"]

        self.model_arch = d["model"]["architecture"]
        self.features = d["model"]["features"]
        self.pixel_window_size = d["model"]["pixel_window_size"]

        self.learning_rate = d["train"]["learning_rate"]
        self.batch_size_train = d["train"]["batch_size"]
        self.batch_size_predict = d["predict"]["batch_size"]
        self.num_epochs = d["train"]["num_epochs"]
        self.label_set = d["input"]["label_set"]
        self.normalization = d["input"]["normalization"]
        self.loss_name = d["train"]["loss"]
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
        if not os.path.exists(self.test_path):
            os.mkdir(self.test_path)
        if not os.path.exists(self.dataset_comparison_path):
            os.mkdir(self.dataset_comparison_path)
        if not os.path.exists(self.meta_data_path):
            os.mkdir(self.meta_data_path)
        if not os.path.exists(self.checkpoints_path):
            os.mkdir(self.checkpoints_path)
        if not os.path.exists(self.plots_path):
            os.mkdir(self.plots_path)

    def load_config(self, path):
        """
        Load configuration from a JSON file.
        :param path: Path to the JSON file.
        """
        with open(path, "rt") as fi:
            self.cfg = json.load(fi)
            # TODO:: Validate config structure

            self.config_from_dict(self.cfg)
            copyfile(path, self.meta_data_path + "/config.json")

    def load_test_products(self):
        """
        Load the list of test products from a text file.
        :return: List of parsed test products.
        """
        test_products_file = open(self.cfg["input"]["test_products"], "r")
        test_products = test_products_file.read().split("\n")
        self.parsed_test_products = []
        for item in test_products:
            if item != "":
                parse = item.split("_")
                index_date = parse[5]+"_"+parse[2]
                self.parsed_test_products.append(index_date)
        return self.parsed_test_products

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
                variable = root.createVariable(name, "f4", dimensions=dimension_names, zlib=True, complevel=9,
                                               endian="little")
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

    def split(self, test_products):
        """
        Read the files in self.path_input_dir, and split them according to self.split_ratio_val, self.split_ratio_test.
        The results are written to output/splits.json.
        """
        self.splits = generate_splits(self.path_data_dir, self.split_ratio_val, test_products)

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
                test_p=100 * len(test_products)
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
        if not os.path.exists(saving_path):
            os.mkdir(saving_path)
        if not os.path.exists(saving_path + "/" + filename_image):
            os.mkdir(saving_path + "/" + filename_image)
        for i, label in enumerate(self.classes):
            saving_filename = saving_path + "/" + filename_image + "/predict_" + label
            current_class = prediction[:, :, i]
            # current_class[current_class >= 0.5] = 255
            # current_class[current_class < 0.5] = 0
            current_class *= 255
            current_class = current_class.astype(np.uint8)
            skio.imsave(saving_filename + ".png", current_class)
        classification = classification * 63 + 3
        classification[classification > 255] = 20
        classification = classification.astype(np.uint8)
        # skio.imsave(saving_path + "/" + filename_image + "/prediction.png", classification)
        im = Image.fromarray(classification)
        im.save(saving_path + "/" + filename_image + "/prediction.png")
        return True

    def to_txt_normalization(self, train_std, train_means, train_min, train_max):
        file = open(self.meta_data_path+"/normalization.txt", "w")
        file.write("Std: "+str(train_std)+"\n"+"Mean: "+str(train_means)+"\n"+"Min: "+str(train_min)+"\n"+"Max: "+str(train_max))
        file.close()

    def to_txt_class_weights(self, weights):
        file = open(self.meta_data_path+"/class_weights.txt", "w")
        file.write(str(weights))
        file.close()

    def set_batches_f1(self, true, predictions, batches):
        samples = len(true) // batches
        f1 = 0
        iteration = 0
        for i in range(batches):
            curr_f1 = self.model.custom_f1(true[i * samples:(i + 1) * samples], predictions[i * samples:(i + 1) * samples])
            f1 += curr_f1
            iteration += 1
        average_f1 = f1 / iteration
        return average_f1

    def set_batches_precision(self, true, predictions, batches):
        samples = len(true) // batches
        f1 = 0
        iteration = 0
        for i in range(batches):
            curr_f1 = self.model.precision_m(true[i * samples:(i + 1) * samples], predictions[i * samples:(i + 1) * samples])
            f1 += curr_f1
            iteration += 1
        average_precision = f1 / iteration
        return average_precision

    def set_batches_recall(self, true, predictions, batches):
        samples = len(true) // batches
        f1 = 0
        iteration = 0
        for i in range(batches):
            curr_f1 = self.model.recall_m(true[i * samples:(i + 1) * samples], predictions[i * samples:(i + 1) * samples])
            f1 += curr_f1
            iteration += 1
        average_recall = f1 / iteration
        return average_recall

    def get_model_memory_usage(self, batch_size, model):
        import numpy as np
        try:
            from keras import backend as K
        except:
            from tensorflow.keras import backend as K

        shapes_mem_count = 0
        internal_model_mem_count = 0
        for l in model.layers:
            layer_type = l.__class__.__name__
            if layer_type == 'Model':
                internal_model_mem_count += self.get_model_memory_usage(batch_size, l)
            single_layer_mem = 1
            out_shape = l.output_shape
            if type(out_shape) is list:
                out_shape = out_shape[0]
            for s in out_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

        number_size = 4.0
        if K.floatx() == 'float16':
            number_size = 2.0
        if K.floatx() == 'float64':
            number_size = 8.0

        total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
        return gbytes

    def train(self, trainer_name='unet', pretrained_weights=False):
        """
        Fit a model to the training dataset (obtained from a splitting operation).
        """
        # Put self.params to function
        self.params["features"] = self.features
        self.params["batch_size"] = self.batch_size_train
        self.params["label_set"] = self.label_set
        self.params["normalization"] = self.normalization
        self.params["product_level"] = self.product_level

        if self.png_iterator:
            self.features = ["TCI_R", "TCI_G", "TCI_B"]
            training_generator = DataGenerator(self.splits['train'], **self.params, png_form=True)
            validation_generator = DataGenerator(self.splits['val'], **self.params, png_form=True)
            validation_generator.set_normalization_file(self.meta_data_path)
        else:
            training_generator = DataGenerator(self.splits['train'], **self.params)
            validation_generator = DataGenerator(self.splits['val'], **self.params)
            validation_generator.set_normalization_file(self.meta_data_path)

        self.get_model_by_name(self.model_arch)
        # Propagate configuration parameters.

        checkpoint_prefix = os.path.abspath(self.checkpoints_path + "/" + trainer_name + "_")
        self.model.set_checkpoint_prefix(checkpoint_prefix)
        self.model.set_num_epochs(self.num_epochs)
        self.model.set_batch_size(self.batch_size_train)
        self.model.set_learning_rate(self.learning_rate)

        # Construct and compile the model.
        self.model.construct(self.dim[0], self.dim[1], len(self.features), len(self.classes), layers=5, units=64,
                             pretrained_weights=pretrained_weights)
        self.model.model.summary()
        self.model.compile(self.loss_name)

        size = self.get_model_memory_usage(self.batch_size_train, self.model.model)

        self.model.set_num_samples(len(self.splits['train']), len(self.splits['val']))

        if not self.png_iterator:
            train_std, train_means, train_min, train_max = set_normalization(training_generator, self.splits['train'],
                                                                             6)
            self.log.info(train_std, train_means, train_min, train_max)
            self.to_txt_normalization(train_std, train_means, train_min, train_max)

        model_name = trainer_name+"-{}".format(int(time.time()))

        # Fit the model, storing weights in checkpoints/.
        history = self.model.fit(training_generator,
                                 validation_generator, model_name)
        draw_history_plots(history, self.experiment_name, self.experiment_res_folder)

    def parameter_tune(self, trainer_name='unet', pretrained_weights=False):
        """
        Tune hyperparameters with tensorboard.
        """
        # Put self.params to function
        self.params["features"] = self.features
        self.params["batch_size"] = self.batch_size_train
        self.params["label_set"] = self.label_set
        self.params["normalization"] = self.normalization
        self.params["product_level"] = self.product_level

        if self.png_iterator:
            self.features = ["TCI_R", "TCI_G", "TCI_B"]
            training_generator = DataGenerator(self.splits['train'], **self.params, png_form=True)
            validation_generator = DataGenerator(self.splits['val'], **self.params, png_form=True)
            validation_generator.set_normalization_file(self.meta_data_path)
        else:
            training_generator = DataGenerator(self.splits['train'], **self.params)
            validation_generator = DataGenerator(self.splits['val'], **self.params)
            validation_generator.set_normalization_file(self.meta_data_path)

        if not self.png_iterator:
            train_std, train_means, train_min, train_max = set_normalization(training_generator, self.splits['train'],
                                                                             6)
            self.log.info(train_std, train_means, train_min, train_max)
            self.to_txt_normalization(train_std, train_means, train_min, train_max)

        self.get_model_by_name(self.model_arch)
        layer_numbers = [5, 6, 7]
        units_per_layer = [32, 64, 128]
        for layer in layer_numbers:
            for units in units_per_layer:
                model_name = "{}-layer_unet-{}-units-{}".format(layer, units, int(time.time()))
                self.get_model_by_name(self.model_arch)
                # Propagate configuration parameters.

                checkpoint_prefix = os.path.abspath(self.checkpoints_path + "/" + trainer_name + "_"+str(layer) + "_"
                                                    + str(units) + "_")
                self.model.set_checkpoint_prefix(checkpoint_prefix)
                self.model.set_num_epochs(self.num_epochs)
                self.model.set_batch_size(self.batch_size_train)
                self.model.set_learning_rate(self.learning_rate)

                # Construct and compile the model.
                self.model.construct(self.dim[0], self.dim[1], len(self.features), len(self.classes), layer, units,
                                     pretrained_weights)
                self.model.model.summary()
                self.model.compile(self.loss_name)

                size = self.get_model_memory_usage(self.batch_size_train, self.model.model)

                self.model.set_num_samples(len(self.splits['train']), len(self.splits['val']))

                # Fit the model, storing weights in checkpoints/.
                history = self.model.fit(training_generator,
                                         validation_generator, model_name)
                #draw_history_plots(history, self.experiment_name, self.experiment_res_folder)

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
        self.model.compile(self.loss_name)

        # Load model weights.
        self.model.load_weights(self.checkpoints_path + "/" + path_weights)

        # Create an array for storing the segmentation mask.
        probabilities = np.zeros((self.dim[0], self.dim[1], len(self.classes)), dtype=np.float)
        class_mask = np.zeros((self.dim[0], self.dim[1]), dtype=np.uint8)
        self.params["features"] = self.features
        self.params["batch_size"] = self.batch_size_predict
        self.params["shuffle"] = False
        self.params["label_set"] = self.label_set
        self.params["normalization"] = self.normalization
        self.params["product_level"] = self.product_level

        # Read splits again
        path_splits = os.path.abspath(self.meta_data_path + "/splits.json")
        with open(path_splits, "r") as fo:
            dictionary = json.load(fo)

        valid_generator = DataGenerator(dictionary['val'], **self.params)
        valid_generator.set_normalization_file(self.meta_data_path)
        # valid_generator.store_orig(dictionary['val'], self.prediction_path)
        # val_std, val_means, val_min, val_max = set_normalization(valid_generator, dictionary['val'], 1)
        # valid_generator.store_orig(dictionary['val'], self.prediction_path)
        length = dictionary['total']
        temp_list = dictionary['filepaths'][0:length // 20]
        # out = valid_generator.get_labels(temp_list, self.prediction_path, self.validation_path, self.classes)
        # for i in range(20):
        #    if i > 0:
        #        temp_list = dictionary['filepaths'][(i*length)//20:((i+1)*length) // 20]
        #        valid_generator.get_labels(temp_list, self.prediction_path, self.validation_path, self.classes)
        predictions = self.model.predict(valid_generator)
        y_pred = np.argmax(predictions, axis=3)
        classes = valid_generator.get_classes()
        y_true = np.argmax(classes, axis=3)

        f1_kmask = np.round(self.set_batches_f1(classes, predictions, 20), 2)

        y_pred_fl = y_pred.flatten()
        y_true_fl = y_true.flatten()
        unique_true = np.unique(y_true_fl)
        cm, cm_normalize, cm_multi, cm_multi_norm = self.model.get_confusion_matrix(y_true_fl, y_pred_fl, self.classes)
        self.log.info(cm_normalize)
        plot_confusion_matrix(cm_normalize, self.classes,
                              "Confusion matrix for KappaMask, dice score: " + str(f1_kmask),
                              normalized=True)
        plt.savefig(os.path.join(self.plots_path, 'confusion_matrix_plot.png'))
        plt.close()

        for i, prediction in enumerate(predictions):
            self.save_masks_contrast(dictionary['val'][i], prediction, y_pred[i], self.prediction_path)

        sen2cor = valid_generator.get_sen2cor(self.prediction_path)
        y_sen2cor = np.argmax(sen2cor, axis=3)
        f1_sen2cor = np.round(self.set_batches_f1(classes, sen2cor, 20), 2)
        y_sen2cor_fl = y_sen2cor.flatten()
        y_true_fl = y_true.flatten()
        unique_true = np.unique(y_true_fl)
        cm, cm_normalize, cm_multi, cm_multi_norm = self.model.get_confusion_matrix(y_true_fl, y_sen2cor_fl,
                                                                                    self.classes)
        self.log.info(cm_normalize)
        plot_confusion_matrix(cm_normalize, self.classes,
                              "Confusion matrix for sen2cor, dice score: " + str(f1_sen2cor),
                              normalized=True)
        plt.savefig(os.path.join(self.plots_path, 'confusion_matrix_sen2cor.png'))
        plt.close()

        """for i, matrix in enumerate(cm_multi_norm):
            plt.figure()
            plot_confusion_matrix(matrix, ["Other", self.classes[i]], self.experiment_name + ": cf_matrix "+self.classes[i], normalized=True)
            plt.savefig(os.path.join(self.plots_path, 'cf_matrix_'+self.classes[i]+'.png'))
            plt.close()"""

        # classes = self.model.predict_classes_gen(valid_generator)

        """self.save_to_nc("output/model_v1/prediction.nc", "probabilities", probabilities)
        self.save_to_img("output/model_v1/prediction.png", class_mask)
        self.save_to_img_contrast("output/model_v1/prediction_contrast.png", class_mask)"""

    def validation(self, datadir, path_weights):
        """
        Validate predicted data
        Create folder with substracted images
        Output metrics
        :param path_original: Path to the input data cube.
        :param path_predicted: Path to the model weights.
        """
        self.get_model_by_name(self.model_arch)

        # Propagate configuration parameters.
        self.model.set_batch_size(self.batch_size_predict)

        # Construct and compile the model.
        self.model.construct(self.dim[0], self.dim[1], len(self.features), len(self.classes))
        self.model.compile(self.loss_name)

        # Load model weights.
        self.model.load_weights(self.checkpoints_path + "/" + path_weights)

        probabilities = np.zeros((self.dim[0], self.dim[1], len(self.classes)), dtype=np.float)
        class_mask = np.zeros((self.dim[0], self.dim[1]), dtype=np.uint8)
        self.params["features"] = self.features
        self.params["batch_size"] = self.batch_size_predict
        self.params["shuffle"] = False
        self.params["label_set"] = self.label_set
        self.params["normalization"] = self.normalization
        self.params["product_level"] = self.product_level

        tile_paths = []

        for subfolder in os.listdir(datadir):
            tile_paths.append(os.path.join(datadir, subfolder))
        test_generator = DataGenerator(tile_paths, **self.params)
        test_generator.set_normalization_file(self.meta_data_path)
        #test_std, test_means, test_min, test_max = set_normalization(test_generator, tile_paths, 1)
        # test_generator.get_labels(tile_paths, self.prediction_path, self.validation_path, self.classes)
        test_generator.store_orig(tile_paths, self.validation_path)

        predictions = self.model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=3)
        for i, prediction in enumerate(predictions):
            self.save_masks_contrast(tile_paths[i], prediction, y_pred[i], self.validation_path)

        """self.save_to_nc("output/model_v1/prediction.nc", "probabilities", probabilities)
        self.save_to_img("output/model_v1/prediction.png", class_mask)
        self.save_to_img_contrast("output/model_v1/prediction_contrast.png", class_mask)"""

    def test(self, product_name, path_weights):

        self.get_model_by_name(self.model_arch)

        # Propagate configuration parameters.
        self.model.set_batch_size(self.batch_size_predict)

        # Construct and compile the model.
        self.model.construct(self.dim[0], self.dim[1], len(self.features), len(self.classes))
        self.model.compile(self.loss_name)

        # Load model weights.
        self.model.load_weights(self.checkpoints_path + "/" + path_weights)

        # Create an array for storing the segmentation mask.
        probabilities = np.zeros((self.dim[0], self.dim[1], len(self.classes)), dtype=np.float)
        class_mask = np.zeros((self.dim[0], self.dim[1]), dtype=np.uint8)
        self.params["features"] = self.features
        self.params["batch_size"] = self.batch_size_predict
        self.params["shuffle"] = False
        self.params["label_set"] = self.label_set
        self.params["normalization"] = self.normalization
        self.params["product_level"] = self.product_level

        path_splits = os.path.abspath(self.meta_data_path + "/splits.json")
        with open(path_splits, "r") as fo:
            dictionary = json.load(fo)

        test_generator = DataGenerator(dictionary['test'], **self.params)
        test_generator.set_normalization_file(self.meta_data_path)
        test_generator.get_labels(dictionary['test'], self.test_path, self.classes)

        predictions = self.model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=3)
        classes = test_generator.get_classes()
        y_true = np.argmax(classes, axis=3)

        f1_kmask = np.round(self.set_batches_f1(classes, predictions, 1), 2)

        y_pred_fl = y_pred.flatten()
        y_true_fl = y_true.flatten()
        unique_true = np.unique(y_true_fl)
        self.log.info("Unique KappaMask {} and unique original {}".format(np.unique(y_pred_fl), unique_true))
        self.log.info("F1 KappaMask {}".format(f1_kmask))
        f1_dic, precision, recall = {}, {}, {}
        for i, label in enumerate(unique_true):
            f1_curr = np.round(self.set_batches_f1(classes[:, :, :, label], predictions[:, :, :, label], 1), 2)
            prec_curr = np.round(self.set_batches_precision(classes[:, :, :, label], predictions[:, :, :, label], 1), 2)
            rec_curr = np.round(self.set_batches_recall(classes[:, :, :, label], predictions[:, :, :, label], 1), 2)
            f1_dic[label] = f1_curr
            precision[label] = prec_curr
            recall[label] = rec_curr
        self.log.info("Kappa {}".format(f1_dic))
        self.log.info("precision {} and recall {}".format(precision, recall))
        file = open(self.plots_path + "/test_f1_scores.txt", "w")
        file.write("KappaMask F1: " + str(f1_dic) + "\n")
        file.write("KappaMask Precision: " + str(precision) + "\n")
        file.write("KappaMask Recall: " + str(recall) + "\n")
        cm, cm_normalize, cm_multi, cm_multi_norm = self.model.get_confusion_matrix(y_true_fl, y_pred_fl, self.classes)
        self.log.info(cm_normalize)
        plot_confusion_matrix(cm_normalize[1:-1, 1:-1], ["CLEAR", "CLOUD_SHADOW", "SEMI_TRANSPARENT_CLOUD", "CLOUD", "MISSING"],
                              "Test confusion matrix for KappaMask, dice score: " + str(f1_kmask),
                              normalized=True, smaller=True)
        plt.savefig(os.path.join(self.plots_path, 'test_confusion_matrix_plot.png'))
        plt.close()

        for i, prediction in enumerate(predictions):
            self.save_masks_contrast(dictionary['test'][i], prediction, y_pred[i], self.test_path)

        """for i, matrix in enumerate(cm_multi_norm):
            plt.figure()
            plot_confusion_matrix(matrix, ["Other", self.classes[i]],
                                  self.experiment_name + ": cf_matrix " + self.classes[i], normalized=True)
            plt.savefig(os.path.join(self.plots_path, 'cf_matrix_' + self.classes[i] + '_test.png'))
            plt.close()"""

        sen2cor = test_generator.get_sen2cor(self.test_path)
        y_sen2cor = np.argmax(sen2cor, axis=3)
        f1_sen2cor = np.round(self.set_batches_f1(classes, sen2cor, 1), 2)
        y_sen2cor_fl = y_sen2cor.flatten()
        y_true_fl = y_true.flatten()
        unique_true = np.unique(y_true_fl)
        self.log.info("Unique Sen2Cor {}".format(np.unique(y_sen2cor_fl)))
        self.log.info("F1 Sen2Cor {}".format(f1_sen2cor))
        f1_dic, precision, recall = {}, {}, {}
        for i, label in enumerate(unique_true):
            f1_curr = np.round(self.set_batches_f1(classes[:, :, :, label], sen2cor[:, :, :, label], 1), 2)
            prec_curr = np.round(self.set_batches_precision(classes[:, :, :, label], sen2cor[:, :, :, label], 1), 2)
            rec_curr = np.round(self.set_batches_recall(classes[:, :, :, label], sen2cor[:, :, :, label], 1), 2)
            f1_dic[label] = f1_curr
            precision[label] = prec_curr
            recall[label] = rec_curr
        self.log.info("Sen2Cor {}".format(f1_dic))
        self.log.info("precision {} and recall {}".format(precision, recall))
        file.write("Sen2Cor F1: " + str(f1_dic) + "\n")
        file.write("Sen2Cor Precision: " + str(precision) + "\n")
        file.write("Sen2Cor Recall: " + str(recall) + "\n")
        cm, cm_normalize, cm_multi, cm_multi_norm = self.model.get_confusion_matrix(y_true_fl, y_sen2cor_fl,
                                                                                    self.classes)
        self.log.info("Sen2Cor {}".format(cm_normalize))
        plot_confusion_matrix(cm_normalize[1:-1, 1:-1], self.classes[1:-1],
                              "Test confusion matrix for sen2cor, dice score: " + str(f1_sen2cor),
                              normalized=True, smaller=True)
        plt.savefig(os.path.join(self.plots_path, 'test_confusion_matrix_sen2cor.png'))
        plt.close()

        fmask = test_generator.get_fmask(self.test_path)
        y_fmask = np.argmax(fmask, axis=3)
        f1_fmask = np.round(self.set_batches_f1(classes, fmask, 1), 2)
        y_fmask_fl = y_fmask.flatten()
        y_true_fl = y_true.flatten()
        unique_true = np.unique(y_true_fl)
        self.log.info("Unique Fmask {}".format(np.unique(y_fmask_fl)))
        self.log.info("F1 Fmask {}".format(f1_fmask))
        f1_dic, precision, recall = {}, {}, {}
        for i, label in enumerate(unique_true):
            f1_curr = np.round(self.set_batches_f1(classes[:, :, :, label], fmask[:, :, :, label], 1), 2)
            prec_curr = np.round(self.set_batches_precision(classes[:, :, :, label], fmask[:, :, :, label], 1), 2)
            rec_curr = np.round(self.set_batches_recall(classes[:, :, :, label], fmask[:, :, :, label], 1), 2)
            f1_dic[label] = f1_curr
            precision[label] = prec_curr
            recall[label] = rec_curr
        self.log.info("Fmask {}".format(f1_dic))
        self.log.info("precision {} and recall {}".format(precision, recall))
        file.write("Fmask F1: " + str(f1_dic) + "\n")
        file.write("Fmask Precision: " + str(precision) + "\n")
        file.write("Fmask Recall: " + str(recall) + "\n")
        cm, cm_normalize, cm_multi, cm_multi_norm = self.model.get_confusion_matrix(y_true_fl, y_fmask_fl,
                                                                                    self.classes)
        self.log.info("Fmask {}".format(cm_normalize))
        plot_confusion_matrix(cm_normalize, self.classes[1:-1],
                              "Test confusion matrix for Fmask, dice score: " + str(f1_fmask),
                              normalized=True, smaller=True)
        plt.savefig(os.path.join(self.plots_path, 'test_confusion_matrix_fmask.png'))
        plt.close()

        s2cloudless = test_generator.get_s2cloudless(self.test_path)
        y_s2cloudless = np.argmax(s2cloudless, axis=3)
        f1_s2cloudless = np.round(self.set_batches_f1(classes, s2cloudless, 1), 2)
        y_s2cloudless_fl = y_s2cloudless.flatten()
        y_true_fl = y_true.flatten()
        unique_true = np.unique(y_true_fl)
        self.log.info("Unique s2cloudless {}".format(np.unique(y_s2cloudless_fl)))
        self.log.info("F1 s2cloudless {}".format(f1_s2cloudless))
        f1_dic, precision, recall = {}, {}, {}
        for i, label in enumerate(unique_true):
            f1_curr = np.round(self.set_batches_f1(classes[:, :, :, label], s2cloudless[:, :, :, label], 1), 2)
            prec_curr = np.round(self.set_batches_precision(classes[:, :, :, label], s2cloudless[:, :, :, label], 1), 2)
            rec_curr = np.round(self.set_batches_recall(classes[:, :, :, label], s2cloudless[:, :, :, label], 1), 2)
            f1_dic[label] = f1_curr
            precision[label] = prec_curr
            recall[label] = rec_curr
        self.log.info("S2cloudless {}".format(f1_dic))
        self.log.info("precision {} and recall".format(precision, recall))
        file.write("S2cloudless F1: " + str(f1_dic) + "\n")
        file.write("S2cloudless Precision: " + str(precision) + "\n")
        file.write("S2cloudless Recall: " + str(recall) + "\n")
        cm, cm_normalize, cm_multi, cm_multi_norm = self.model.get_confusion_matrix(y_true_fl, y_s2cloudless_fl,
                                                                                    self.classes)
        self.log.info("s2cloudless {}".format(cm_normalize))
        plot_confusion_matrix(cm_normalize, self.classes[1:-1],
                              "Test confusion matrix for S2cloudless, dice score: " + str(f1_s2cloudless),
                              normalized=True, smaller=True)
        plt.savefig(os.path.join(self.plots_path, 'test_confusion_matrix_s2cloudless.png'))
        plt.close()

        maja = test_generator.get_maja(self.test_path)
        y_maja = np.argmax(maja, axis=3)
        f1_maja = np.round(self.set_batches_f1(classes, maja, 1), 2)
        y_maja_fl = y_maja.flatten()
        y_true_fl = y_true.flatten()
        unique_true = np.unique(y_true_fl)
        self.log.info("Unique maja {}".format(np.unique(y_maja_fl)))
        self.log.info("F1 maja {}".format(f1_maja))
        f1_dic, precision, recall = {}, {}, {}
        for i, label in enumerate(unique_true):
            f1_curr = np.round(self.set_batches_f1(classes[:, :, :, label], maja[:, :, :, label], 1), 2)
            prec_curr = np.round(self.set_batches_precision(classes[:, :, :, label], maja[:, :, :, label], 1), 2)
            rec_curr = np.round(self.set_batches_recall(classes[:, :, :, label], maja[:, :, :, label], 1), 2)
            f1_dic[label] = f1_curr
            precision[label] = prec_curr
            recall[label] = rec_curr
        self.log.info("Maja {}".format(f1_dic))
        self.log.info("precision {} and recall {}".format(precision, recall))
        file.write("Maja F1: " + str(f1_dic) + "\n")
        file.write("Maja Precision: " + str(precision) + "\n")
        file.write("Maja Recall: " + str(recall) + "\n")
        cm, cm_normalize, cm_multi, cm_multi_norm = self.model.get_confusion_matrix(y_true_fl, y_maja_fl,
                                                                                    self.classes)
        self.log.info(cm_normalize)
        plot_confusion_matrix(cm_normalize[1:-1, 1:-1], self.classes[1:-1],
                              "Test confusion matrix for MAJA, dice score: " + str(f1_maja),
                              normalized=True, smaller=True)
        plt.savefig(os.path.join(self.plots_path, 'test_confusion_matrix_maja.png'))
        plt.close()

        return

    def dataset_comparison(self, path_weights):
        self.get_model_by_name(self.model_arch)

        # Propagate configuration parameters.
        self.model.set_batch_size(self.batch_size_predict)

        # Construct and compile the model.
        self.model.construct(self.dim[0], self.dim[1], len(self.features), len(self.classes))
        self.model.compile(self.loss_name)

        # Load model weights.
        self.model.load_weights(self.checkpoints_path + "/" + path_weights)

        # Create an array for storing the segmentation mask.
        probabilities = np.zeros((self.dim[0], self.dim[1], len(self.classes)), dtype=np.float)
        class_mask = np.zeros((self.dim[0], self.dim[1]), dtype=np.uint8)
        self.params["features"] = self.features
        self.params["batch_size"] = self.batch_size_predict
        self.params["shuffle"] = False
        self.params["label_set"] = self.label_set
        self.params["normalization"] = self.normalization

        all_indices = [f for f in os.listdir(self.path_data_dir) if
                       os.path.isfile(os.path.join(self.path_data_dir, f)) and os.path.join(self.path_data_dir, f).endswith('.nc')]
        all_indices_fullname = [os.path.join(self.path_data_dir, index) for index in all_indices]
        data_generator = DataGenerator(all_indices_fullname, **self.params)
        data_generator.set_normalization_file(self.meta_data_path)
        data_generator.get_labels(all_indices_fullname, self.dataset_comparison_path, self.classes)
        predictions = self.model.predict(data_generator)
        y_pred = np.argmax(predictions, axis=3)
        for i, prediction in enumerate(predictions):
            self.save_masks_contrast(all_indices_fullname[i], prediction, y_pred[i], self.dataset_comparison_path)
        classes = data_generator.get_classes()
        y_true = np.argmax(classes, axis=3)

        f1_kmask = np.round(self.set_batches_f1(classes, predictions, 1), 2)

        y_pred_fl = y_pred.flatten()
        y_true_fl = y_true.flatten()
        unique_true = np.unique(y_true_fl)
        self.log.info("Unique KappaMask {} and unique original {}".format(np.unique(y_pred_fl), unique_true))
        self.log.info("F1 KappaMask {}".format(f1_kmask))
        f1_dic, precision, recall = {}, {}, {}
        for i, label in enumerate(unique_true):
            f1_curr = np.round(self.set_batches_f1(classes[:, :, :, label], predictions[:, :, :, label], 1), 2)
            prec_curr = np.round(self.set_batches_precision(classes[:, :, :, label], predictions[:, :, :, label], 1), 2)
            rec_curr = np.round(self.set_batches_recall(classes[:, :, :, label], predictions[:, :, :, label], 1), 2)
            f1_dic[label] = f1_curr
            precision[label] = prec_curr
            recall[label] = rec_curr
        self.log.info("Kappa {}".format(f1_dic))
        self.log.info("precision {} and recall {}".format(precision, recall))
        file = open(self.plots_path + "/dataset_comparison.txt", "w")
        file.write("KappaMask F1: " + str(f1_dic) + " for " + self.label_set + "\n")
        file.write("KappaMask Precision: " + str(precision) + "\n")
        file.write("KappaMask Recall: " + str(recall) + "\n")
        cm, cm_normalize, cm_multi, cm_multi_norm = self.model.get_confusion_matrix(y_true_fl, y_pred_fl, self.classes)
        self.log.info(cm_normalize)
        plot_confusion_matrix(cm_normalize, ["CLEAR", "CLOUD_SHADOW", "SEMI_TRANSPARENT_CLOUD", "CLOUD", "MISSING"],
                              "Confusion matrix " + self.label_set + " for KappaMask, dice score: " + str(f1_kmask),
                              normalized=True, smaller=True)
        plt.savefig(os.path.join(self.plots_path, 'confusion_matrix_' + self.label_set + '.png'))
        plt.close()

        sen2cor = data_generator.get_sen2cor(self.dataset_comparison_path)

        f1_sen2cor = np.round(self.set_batches_f1(classes, sen2cor, 1), 2)
        y_sen2cor = np.argmax(sen2cor, axis=3)
        y_sen2cor_fl = y_sen2cor.flatten()
        y_true_fl = y_true.flatten()
        unique_true = np.unique(y_true_fl)
        self.log.info("Unique Sen2Cor {}".format(np.unique(y_sen2cor_fl)))
        self.log.info("F1 Sen2Cor {}".format(f1_sen2cor))
        f1_dic, precision, recall = {}, {}, {}
        for i, label in enumerate(unique_true):
            f1_curr = np.round(self.set_batches_f1(classes[:, :, :, label], sen2cor[:, :, :, label], 1), 2)
            prec_curr = np.round(self.set_batches_precision(classes[:, :, :, label], sen2cor[:, :, :, label], 1), 2)
            rec_curr = np.round(self.set_batches_recall(classes[:, :, :, label], sen2cor[:, :, :, label], 1), 2)
            f1_dic[label] = f1_curr
            precision[label] = prec_curr
            recall[label] = rec_curr
        self.log.info("Sen2Cor {}".format(f1_dic))
        self.log.info("precision {} and recall {}".format(precision, recall))
        file.write("Sen2Cor F1: " + str(f1_dic) + " for " + self.label_set + "\n")
        file.write("Sen2Cor Precision: " + str(precision) + "\n")
        file.write("Sen2Cor Recall: " + str(recall) + "\n")
        cm, cm_normalize, cm_multi, cm_multi_norm = self.model.get_confusion_matrix(y_true_fl, y_sen2cor_fl, self.classes)
        self.log.info(cm_normalize)
        plot_confusion_matrix(cm_normalize[1:-1, 1:-1], ["CLEAR", "CLOUD_SHADOW", "SEMI_TRANSPARENT_CLOUD", "CLOUD", "MISSING"],
                              "Confusion matrix " + self.label_set + " for Sen2Cor, dice score: " + str(f1_kmask),
                              normalized=True, smaller=True)
        plt.savefig(os.path.join(self.plots_path, 'Sen2Cor_confusion_matrix_' + self.label_set + '.png'))
        plt.close()
        file.close()

    def selecting(self, product_name, path_weights):

        self.get_model_by_name(self.model_arch)

        # Propagate configuration parameters.
        self.model.set_batch_size(self.batch_size_predict)

        # Construct and compile the model.
        self.model.construct(self.dim[0], self.dim[1], len(self.features), len(self.classes))
        self.model.compile(self.loss_name)

        # Load model weights.
        self.model.load_weights(path_weights)

        # Create an array for storing the segmentation mask.
        probabilities = np.zeros((self.dim[0], self.dim[1], len(self.classes)), dtype=np.float)
        class_mask = np.zeros((self.dim[0], self.dim[1]), dtype=np.uint8)
        self.params["features"] = self.features
        self.params["batch_size"] = self.batch_size_predict
        self.params["shuffle"] = False
        self.params["label_set"] = self.label_set
        self.params["normalization"] = self.normalization
        self.params["product_level"] = self.product_level

        file_specificator = product_name.rsplit('.', 1)[0]
        date_match = file_specificator.rsplit('_', 1)[-1]
        index_match = file_specificator.rsplit('_', 1)[0].rsplit('_', 1)[-1]

        # Read splits again
        path_splits = os.path.abspath(self.meta_data_path + "/splits.json")
        with open(path_splits, "r") as fo:
            dictionary = json.load(fo)

        test_generator = DataGenerator(dictionary['val'], **self.params)
        # test_std, test_means, test_min, test_max = set_normalization(test_generator, dictionary['val'], 30)
        # test_generator.get_labels(tile_paths, self.prediction_path, self.validation_path, self.classes)
        test_generator.store_orig(dictionary['val'], self.prediction_path)

        predictions = self.model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=3)
        for i, prediction in enumerate(predictions):
            self.save_masks_contrast(dictionary['val'][i], prediction, y_pred[i], self.prediction_path)
        return

    def run_stats(self):
        self.params["features"] = self.features
        self.params["batch_size"] = self.batch_size_train
        self.params["label_set"] = self.label_set
        self.params["normalization"] = self.normalization
        self.params["product_level"] = self.product_level

        path_splits = os.path.abspath(self.meta_data_path + "/splits.json")
        with open(path_splits, "r") as fo:
            dictionary = json.load(fo)

        training_generator = DataGenerator(dictionary['train'], **self.params)
        #validation_generator = DataGenerator(dictionary['val'], **self.params)
        #test_generator = DataGenerator(dictionary['test'], **self.params)
        train_std, train_means, train_min, train_max = set_normalization(training_generator, dictionary['train'], 5)
        self.to_txt_normalization(train_std, train_means, train_min, train_max)
        self.log.info("Stats")
        self.log.info(train_std, train_means, train_min, train_max)
        tr_overall_stat, tr_per_image_stat = training_generator.get_label_stat(dictionary['train'], self.classes)
        self.to_txt_class_weights(tr_overall_stat)
        #val_overall_stat, val_per_image_stat = validation_generator.get_label_stat(dictionary['val'], self.classes)
        #test_overall_stat, test_per_image_stat = test_generator.get_label_stat(dictionary['test'], self.classes)
        #print(tr_overall_stat, val_overall_stat, test_overall_stat)
        #print(test_per_image_stat)
        draw_4lines(tr_per_image_stat, "train", self.experiment_res_folder, self.classes)
        #draw_4lines(val_per_image_stat, "val", self.experiment_res_folder, self.classes)
        #draw_4lines(test_per_image_stat, "test", self.experiment_res_folder, self.classes)
        return

    def get_origin_im(self):
        self.params["features"] = self.features
        self.params["batch_size"] = self.batch_size_train
        self.params["label_set"] = self.label_set
        self.params["normalization"] = self.normalization
        self.params["product_level"] = self.product_level

        path_splits = os.path.abspath(self.meta_data_path + "/splits.json")
        with open(path_splits, "r") as fo:
            dictionary = json.load(fo)

        train_generator = DataGenerator(dictionary['train'], **self.params)
        train_generator.store_orig(dictionary['train'], self.prediction_path)
        validation_generator = DataGenerator(dictionary['val'], **self.params)
        validation_generator.store_orig(dictionary['val'], self.prediction_path)

        return
