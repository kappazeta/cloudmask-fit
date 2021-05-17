# vim: set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab

# Copyright 2020-2021 KappaZeta Ltd.
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
import numpy as np
import netCDF4 as nc
import tensorflow as tf
from keras.utils import np_utils


from .dg_base import DataGenerator


class DataGeneratorPixelNHood(DataGenerator):
    def __init__(self, list_indices, path_input, batch_size, features, dim, num_classes, label_set, normalization,
                 test_mode=False, shuffle=True, png_form=False, test_products_list=False, pixel_window_size=5,
                 subsample_factor=1):
        super(DataGeneratorPixelNHood, self).__init__(
            list_indices, path_input, batch_size, features, dim, num_classes, label_set, normalization,
            test_mode=test_mode, shuffle=shuffle, png_form=png_form, test_products_list=test_products_list
        )

        self.pixel_window_size = pixel_window_size
        self.subsample_factor = subsample_factor

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch = [self.list_indices[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(batch)

        return X, Y

    def __data_generation(self, list_indices_temp):
        """Generates data containing batch_size samples"""

        # Image width, height.
        w, h = self.dim
        num_pixels = int(w * h * self.subsample_factor)
        # Window width, height.
        ww = self.pixel_window_size
        wh = ww
        # Half-width, half-height of the window.
        hw = int(ww / 2)
        hh = int(wh / 2)

        X = np.zeros((self.batch_size * num_pixels, ww, wh, len(self.features)), dtype=np.float32)
        # Y = np.zeros((self.batch_size * num_pixels, ww, wh, self.num_classes), dtype=np.float32)
        Y = np.zeros((self.batch_size * num_pixels, self.num_classes), dtype=np.float32)

        idx = 0
        # Iterate over NetCDF subtiles.
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith('.nc'):
                # Is the subtile in the list of test products?
                # Skip it from training, then.
                filename = file.split("/")[-1]
                filename_sub = filename.split("_")
                filename_sub = filename_sub[0] + "_" + filename_sub[1]
                if self.test_products_list:
                    if self.test_mode:
                        if filename_sub not in self.test_products_list:
                            continue
                    else:
                        if filename_sub in self.test_products_list:
                            continue

                # Load the subtile.
                with nc.Dataset(file, 'r') as root:
                    # Normalize the features.
                    if self.png_form:
                        data_bands = [(np.asarray(root[f]) / 255.0) for i, f in enumerate(self.features)]
                    else:
                        if self.normalization == "minmax":
                            data_bands = [(np.asarray(root[f]) - self.min_v[i]) / (self.max_v[i]-self.min_v[i]) for i, f in
                                          enumerate(self.features)]
                        else:
                            data_bands = [(np.asarray(root[f]) - self.means[i]) / (self.stds[i]) for i, f
                                          in enumerate(self.features)]

                    label_cat = None
                    # Get labels.
                    try:
                        label = np.asarray(root[self.label_set])
                        label_cat = np_utils.to_categorical(label, self.num_classes)
                    except:
                        print("Label for " + file + " not found")
                        print(data_bands[0].shape)

                    # Stack the features together.
                    data_bands = np.stack(data_bands)
                    # Move the feature axis to last.
                    data_bands = np.rollaxis(data_bands, 0, 3)

                    # Get a list of pixel coordinates.
                    pixel_coords = list(np.ndindex((w, h)))
                    # Random-sample a specific number of pixels from it.
                    pixel_indices = np.random.choice(range(w * h), num_pixels)

                    # Loop over pixels.
                    for pidx in pixel_indices:
                        j, k = pixel_coords[pidx]
                        # Clamped pixel coordinates for the neighbourhood (x0 ... x1, y0 ... y1),
                        # and number of pixels missing due to image edges:
                        #   xs from the left, xe from the right
                        #   ys from the top, ye from the bottom
                        x0, x1, xs, xe = np.array((j - hw, j + hw + 1, hw - j, j + hw - w + 1)).clip(0, w)
                        y0, y1, ys, ye = np.array((k - hh, k + hh + 1, hh - k, k + hh - h + 1)).clip(0, h)
                        # Only copy the pixels with values, and leave pixels exceeding image dimensions as 0.
                        X[idx, xs:(ww - xe), ys:(wh - ye)] = data_bands[x0:x1, y0:y1, :]
                        # Copy labels the same way, if applicable.
                        if label_cat is not None:
                            # Y[idx, xs:(ww - xe), ys:(wh - ye)] = label_cat[x0:x1, y0:y1, :]
                            Y[idx] = label_cat[j, k, :]
                        idx += 1

        return X, Y
