import numpy as np
import netCDF4 as nc
import os
import random


class DataGenerator(object):
    def __init__(self, path_input):
        self.path = path_input
        self.indices = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f)) and os.path.join(self.path, f).endswith('.nc')]
        self.total_length = len(self.indices)

    def split(self, ratio_val, ratio_test):
        self.ratio_val = ratio_val
        self.ratio_test = ratio_test
        validation_test_number = int((self.ratio_val + self.ratio_test) * self.total_length)
        test_number = int(self.ratio_test * self.total_length)
        val_test_indices = random.sample(self.indices, validation_test_number)
        self.test_indices = random.sample(val_test_indices, test_number)
        self.val_indices = list(set(val_test_indices) - set(self.test_indices))
        self.train_indices = list(set(self.indices) - set(val_test_indices))

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]