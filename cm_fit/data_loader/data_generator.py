import numpy as np
import netCDF4 as nc
import os
import random
import keras
import skimage.io as skio
from keras.utils import np_utils
from PIL import Image
from tensorflow.python.keras.utils.data_utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, list_indices, path_input, batch_size, features, dim, num_classes, shuffle=True):
        """ Initialization """
        self.path = path_input
        self.stds = [0.00044, 0.037, 0.035, 0.034, 0.035, 0.033, 0.035, 0.033, 0.025, 0.021, 0.0049]
        self.means = [0.0010, 0.02, 0.02, 0.02, 0.025, 0.033, 0.038, 0.038, 0.03, 0.022, 0.01]
        self.min_v = []
        self.max_v = []
        self.list_indices = list_indices
        self.total_length = len(self.list_indices)
        self.batch_size = batch_size
        self.features = features
        self.dim = dim
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_indices) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch = [self.list_indices[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch)

        return X, y

    def set_std(self, stds):
        self.stds = stds

    def set_means(self, means):
        self.means = means

    def set_min(self, min_v):
        self.min_v = min_v

    def set_max(self, max_v):
        self.max_v = max_v

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_indices))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_label_stat(self, list_indices_temp, classes):
        overall_stat = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        per_image_stat = []
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith('.nc'):
                with nc.Dataset(file, 'r') as root:
                    try:
                        label = np.asarray(root['Label'])
                        unique_elements, counts_elements = np.unique(label, return_counts=True)
                        curr_dic = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                        for j in range(len(unique_elements)):
                            overall_stat[unique_elements[j]] += counts_elements[j]
                            curr_dic[j] += counts_elements[j]
                        curr_dic = {k: v / sum(curr_dic.values()) for k, v in curr_dic.items()}
                        per_image_stat.append(curr_dic)
                    except:
                        print("Label for " + file + " not found")
                    # img = Image.fromarray(data_bands, 'RGB')
                    file_name = file.split(".")[0].split("/")[-1]
                    # img.save(path_prediction+"/"+file_name+"orig.png")

        overall_stat = {k: v / sum(overall_stat.values()) for k, v in overall_stat.items()}
        return overall_stat, per_image_stat

    def get_labels(self, list_indices_temp, path_prediction, path_val, classes):
        """Save labels to folder"""
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith('.nc'):
                with nc.Dataset(file, 'r') as root:
                    y = np.empty((self.dim[0], self.dim[1], self.num_classes), dtype=int)
                    data_bands = [np.asarray(root[f])
                                  for i, f in enumerate(["TCI_R", "TCI_G", "TCI_B"])]

                    # data_bands = [(np.asarray(root[f]) - self.min_v[i + 1]) / (self.max_v[i + 1]-self.min_v[i + 1])
                    #              for i, f in enumerate(["B02", "B03", "B04"])]
                    data_bands = np.stack(data_bands)
                    # data_bands /= np.max(np.abs(data_bands), axis=0)
                    # data_bands = (data_bands-np.min(data_bands))/\
                    #             (np.max(data_bands)-np.min(data_bands))
                    # data_bands *= 255.0
                    # data_bands *= (255.0/(np.max(np.abs(data_bands))))
                    data_bands = np.rollaxis(data_bands, 0, 3)
                    try:
                        label = np.asarray(root['Label'])
                        y = np_utils.to_categorical(label, self.num_classes)
                        sen2cor_cc = np.asarray(root['S2CC'])
                        sen2cor_cs = np.asarray(root['S2CS'])
                        sen2cor_scl = np.asarray(root['SCL'])
                    except:
                        print("Label for " + file + " not found")
                        print(data_bands[0].shape)
                    # img = Image.fromarray(data_bands, 'RGB')
                    file_name = file.split(".")[0].split("/")[-1]
                    # img.save(path_prediction+"/"+file_name+"orig.png")

                    if not os.path.exists(path_prediction + "/" + file_name):
                        os.mkdir(path_prediction + "/" + file_name)
                    if not os.path.exists(path_val + "/" + file_name):
                        os.mkdir(path_val + "/" + file_name)

                    # Lossy conversion Range [-0.5882352590560913, 6.766853332519531].
                    unique_before = np.unique(data_bands)
                    # data_bands *= 255
                    data_bands = data_bands.astype(np.uint8)
                    skio.imsave(path_prediction + "/" + file_name + "/orig.png", data_bands)

                    label = label * 63 + 3
                    label = label.astype(np.uint8)
                    # skio.imsave(saving_path + "/" + filename_image + "/prediction.png", classification)
                    im = Image.fromarray(label)
                    im.save(path_prediction + "/" + file_name + "/label.png")

                    # sen2cor_cc = sen2cor_cc.astype(np.uint8)
                    # sen2cor_cs = sen2cor_cs.astype(np.uint8)
                    # sen2cor_cs *= 255
                    sen2cor_scl = sen2cor_scl * 63 + 3

                    sen2cor_scl = sen2cor_scl.astype(np.uint8)
                    # skio.imsave(saving_path + "/" + filename_image + "/prediction.png", classification)
                    im = Image.fromarray(sen2cor_scl)
                    im.save(path_prediction + "/" + file_name + "/SCL.png")

                    skio.imsave(path_prediction + "/" + file_name + "/S2CC.png", sen2cor_cc)
                    for j, curr_cl in enumerate(classes):
                        saving_filename = path_prediction + "/" + file_name + "/" + curr_cl
                        curr_array = y[:, :, j].copy()
                        curr_array *= 255
                        curr_array = curr_array.astype(np.uint8)
                        skio.imsave(saving_filename + ".png", curr_array)

    def get_normal_par(self, list_indices_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        X = np.empty((len(list_indices_temp), self.dim[0], self.dim[1], len(self.features)))
        # Initialization
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith('.nc'):
                with nc.Dataset(file, 'r') as root:
                    data_bands = [np.asarray(root[f]) for f in self.features]
                    try:
                        label = np.asarray(root['Label'])
                    except:
                        print("Label for " + file + " not found")
                        print(data_bands[0].shape)
                    data_bands = np.stack(data_bands)
                    data_bands = np.rollaxis(data_bands, 0, 3)
                    # data_bands = data_bands.reshape((self.dim[0], self.dim[1], len(self.features)))
                    X[i,] = data_bands

        stds_list = []
        means_list = []
        unique_list = []
        min_list = []
        max_list = []
        X_reshaped = np.reshape(X, (len(list_indices_temp) * self.dim[0] * self.dim[1], len(self.features)))
        for j, class_curr in enumerate(self.features):
            # print(class_curr)
            std_array = np.std(X_reshaped[:, j])
            mean_array = np.mean(X_reshaped[:, j])
            unique = np.unique(X_reshaped[:, j])
            min_ar = np.min(X_reshaped[:, j])
            max_ar = np.max(X_reshaped[:, j])
            stds_list.append(std_array)
            means_list.append(mean_array)
            unique_list.append(unique)
            min_list.append(min_ar)
            max_list.append(max_ar)

        return stds_list, means_list, unique_list, min_list, max_list

    def __data_generation(self, list_indices_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], len(self.features)))
        y = np.empty((self.batch_size, self.dim[0], self.dim[1], self.num_classes), dtype=int)
        # Initialization
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith('.nc'):
                with nc.Dataset(file, 'r') as root:
                    data_bands = [(np.asarray(root[f]) - self.means[i]) / self.stds[i] for i, f in
                                  enumerate(self.features)]
                    try:
                        label = np.asarray(root['Label'])
                        unique_lbl = np.unique(label)

                        y[i] = np_utils.to_categorical(label, self.num_classes)
                    except:
                        print("Label for " + file + " not found")
                        print(data_bands[0].shape)
                    data_bands = np.stack(data_bands)
                    data_bands = np.rollaxis(data_bands, 0, 3)
                    # data_bands = data_bands.reshape((self.dim[0], self.dim[1], len(self.features)))
                    X[i,] = data_bands

        return X, y

    def get_classes(self):
        y = np.empty((len(self.list_indices), self.dim[0], self.dim[1], self.num_classes), dtype=int)
        # Initialization
        for i, file in enumerate(self.list_indices):
            if os.path.isfile(file) and file.endswith('.nc'):
                with nc.Dataset(file, 'r') as root:
                    try:
                        label = np.asarray(root['Label'])
                        y[i] = np_utils.to_categorical(label, self.num_classes)
                    except:
                        print("Label for " + file + " not found")
        return y


class TestDataGenerator(keras.utils.Sequence):
    def __init__(self, list_indices, path_input, batch_size, features, dim, shuffle=False):
        """ Initialization """
        self.path = path_input
        self.list_indices = list_indices
        self.total_length = len(self.list_indices)
        self.batch_size = batch_size
        self.features = features
        self.dim = dim
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_indices) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch = [self.list_indices[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation_predict(batch)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_indices))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation_predict(self, list_indices_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], len(self.features)))
        y = np.empty((self.batch_size, self.dim[0], self.dim[1]), dtype=int)
        # Initialization
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith('.nc'):
                with nc.Dataset(file, 'r') as root:
                    data_bands = [np.asarray(root[f]) for f in self.features]
                    try:
                        label = np.asarray(root['Label'])
                        y[i] = label
                        # data_bands = np.stack(data_bands)
                        # data_bands = data_bands.reshape((self.dim[0], self.dim[1], len(self.features)))
                        # X[i,] = data_bands
                    except:
                        print("Label for " + file + " not found")
                        print(data_bands[0].shape)
                        # y[i] = np.zeros_like(data_bands[0])
                    data_bands = np.stack(data_bands)
                    data_bands = data_bands.reshape((self.dim[0], self.dim[1], len(self.features)))
                    X[i,] = data_bands
        return X, y
