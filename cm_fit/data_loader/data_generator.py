import numpy as np
import netCDF4 as nc
import os
import random
import keras
from keras.utils import np_utils


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_indices, path_input, batch_size, features, dim, num_classes, shuffle=True):
        """ Initialization """
        self.path = path_input
        self.stds = [0.00044, 0.037, 0.035, 0.034, 0.035, 0.033, 0.035, 0.033, 0.025, 0.021, 0.0049]
        self.means = [0.0010, 0.02, 0.02, 0.02, 0.025, 0.033, 0.038, 0.038, 0.03, 0.022, 0.01]
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


    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_indices))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

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
        X_reshaped = np.reshape(X, (len(list_indices_temp)*self.dim[0]*self.dim[1], len(self.features)))
        for j, class_curr in enumerate(self.features):
            #print(class_curr)
            std_array = np.std(X_reshaped[:, j])
            mean_array = np.mean(X_reshaped[:, j])
            unique = np.unique(X_reshaped[:, j])
            stds_list.append(std_array)
            means_list.append(mean_array)
            unique_list.append(unique)

        return stds_list, means_list, unique_list

    def __data_generation(self, list_indices_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], len(self.features)))
        y = np.empty((self.batch_size, self.dim[0], self.dim[1], self.num_classes), dtype=int)
        # Initialization
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith('.nc'):
                with nc.Dataset(file, 'r') as root:
                    data_bands = [(np.asarray(root[f]) - self.means[i])/self.stds[i] for i, f in enumerate(self.features)]
                    try:
                        label = np.asarray(root['Label'])
                        y[i] = np_utils.to_categorical(label, self.num_classes)
                    except:
                        print("Label for " + file + " not found")
                        print(data_bands[0].shape)
                    data_bands = np.stack(data_bands)
                    data_bands = np.rollaxis(data_bands, 0, 3)
                    # data_bands = data_bands.reshape((self.dim[0], self.dim[1], len(self.features)))
                    X[i,] = data_bands

        return X, y


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
                        #data_bands = np.stack(data_bands)
                        #data_bands = data_bands.reshape((self.dim[0], self.dim[1], len(self.features)))
                        #X[i,] = data_bands
                    except:
                        print("Label for " + file + " not found")
                        print(data_bands[0].shape)
                        #y[i] = np.zeros_like(data_bands[0])
                    data_bands = np.stack(data_bands)
                    data_bands = data_bands.reshape((self.dim[0], self.dim[1], len(self.features)))
                    X[i,] = data_bands
        return X, y


