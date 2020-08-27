import numpy as np
import netCDF4 as nc
import os
import random


# use https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly for structure
class DataGenerator(object):
    def __init__(self, list_indices, path_input, batch_size, features, dim, shuffle=True):
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
        list_indices_temp = [self.list_indices[k] for k in indexes]

        # Generate data
        data, label = self.__data_generation(list_indices_temp)

        return data, label

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_indices))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generate(self, list_indices_temp):
        # Initialization
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith('.nc'):
                with nc.Dataset(file, 'r') as root:
                    data_bands = [np.asarray(root[f]) for f in self.features]
                    data_bands = np.stack(data_bands)
                    try:
                        label = np.asarray(root['Label'])
                    except:
                        print("Label for " + file + " not found")

        return data_bands, label

    def __data_generation(self, list_indices_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        data = np.empty((self.batch_size, self.dim, self.features))
        labels = np.empty(self.batch_size, dtype=int)
        # Initialization
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith('.nc'):
                with nc.Dataset(file, 'r') as root:
                    data_bands = [np.asarray(root[f]) for f in self.features]
                    data_bands = np.stack(data_bands)
                    print(data_bands.shape)
                    try:
                        label = np.asarray(root['Label'])
                        labels[i] = label
                    except:
                        print("Label for " + file + " not found")
                        print(data_bands[0].shape)
                        labels[i] = np.zeros_like(data_bands[0])
                    data[i] = data_bands

        return data, labels
