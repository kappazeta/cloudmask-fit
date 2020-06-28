# vim: set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab

# Copyright 2020 KappaZeta Ltd.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import numpy as np
import netCDF4 as nc


class CMBGenerator(object):
    @staticmethod
    def shape(path_input):
        fpath = path_input
        if os.path.isfile(fpath) and fpath.endswith('.nc'):
            with nc.Dataset(fpath, 'r') as root:
                band_names = list(root.variables.keys())
                return np.asarray(root[band_names[0]]).shape

    @staticmethod
    def split(path_input_dir, ratio_val=0.1, ratio_test=0.1):
        # String arguments become bytes.
        if isinstance(path_input_dir, bytes):
            path_input_dir = path_input_dir.decode()

        filepaths = []
        dataset = []

        for fname in os.listdir(path_input_dir):
            path_input_file = os.path.join(path_input_dir, fname)

            if os.path.isfile(path_input_file) and fname.endswith('.nc'):
                file_index = len(filepaths)
                filepaths.append(path_input_file)

                with nc.Dataset(path_input_file, 'r') as root:
                    labels = np.asarray(root['Label'])

                    coords = np.where(labels > 0)
                    for i in range(coords[0].shape[0]):
                        x = coords[0][i]
                        y = coords[1][i]

                        dataset.append((file_index, x, y))

        rng = np.random.default_rng()
        rng.shuffle(dataset)

        idx_test_start = 0
        idx_test_end = int(len(dataset) * ratio_test)
        idx_val_start = idx_test_end + 1
        idx_val_end = idx_val_start + int(len(dataset) * ratio_val)
        idx_train_start = idx_val_end + 1

        d = {
            'total': len(dataset),
            'filepaths': filepaths,
            'test': dataset[idx_test_start:idx_test_end],
            'val': dataset[idx_val_start:idx_val_end],
            'train': dataset[idx_train_start:]
        }

        return d

    @staticmethod
    def generator_train(dataset_splits, name_split, features, pixel_window_size, batch_size, num_epochs, num_classes):
        # Extent of the pixel window beyond the central pixel of the sample.
        hwin = (pixel_window_size - 1) >> 1

        # Convert from bytes back into strings.
        name_split = name_split.decode()
        features = [f.decode() for f in features]

        # Unpickle a dict of lists with strings and integers.
        dataset_splits = pickle.loads(dataset_splits)

        filepaths = dataset_splits['filepaths']

        for e in range(num_epochs):
            for i in range(len(filepaths)):
                fpath = filepaths[i]
                if os.path.isfile(fpath) and fpath.endswith('.nc'):
                    with nc.Dataset(fpath, 'r') as root:
                        data_bands = {f: np.asarray(root[f]) for f in features}
                        labels = np.asarray(root['Label'])
                        width, height = labels.shape

                        coords = dataset_splits[name_split]

                        samples_p = []
                        samples_x = []
                        samples_y = []
                        for file_index, x, y in coords:
                            if file_index != i:
                                continue

                            # TODO:: Batches across files
                            # TODO:: Filled batches
                            # TODO:: Padded samples

                            if hwin < x < width - hwin and hwin < y < height - hwin:
                                label = labels[x, y]

                                bands = []
                                for f in features:
                                    band = data_bands[f][(x - hwin):(x + hwin + 1), (y - hwin):(y + hwin + 1)]
                                    bands.append(band)

                                sample = np.stack(bands)
                                # Move features from the first axis to the last axis.
                                sample = np.rollaxis(sample, 0, 3)

                                samples_p.append((x, y))
                                samples_x.append(sample)
                                samples_y.append(np.identity(num_classes)[label])

                            if len(samples_x) == batch_size:
                                batch_p = np.stack(samples_p)
                                batch_x = np.stack(samples_x)
                                batch_y = np.stack(samples_y)

                                samples_p = []
                                samples_x = []
                                samples_y = []

                                yield batch_x, batch_y

    @staticmethod
    def generator_predict(path_input, features, pixel_window_size, batch_size):
        # Extent of the pixel window beyond the central pixel of the sample.
        hwin = (pixel_window_size - 1) >> 1

        # Convert from bytes back into strings.
        if isinstance(path_input, bytes):
            path_input = path_input.decode()
        if isinstance(features[0], bytes):
            features = [f.decode() for f in features]

        fpath = path_input
        if os.path.isfile(fpath) and fpath.endswith('.nc'):
            with nc.Dataset(fpath, 'r') as root:
                data_bands = {f: np.asarray(root[f]) for f in features}
                band_names = list(data_bands.keys())
                width, height = data_bands[band_names[0]].shape

                labels = None
                if 'Label' in root.variables.keys():
                    labels = np.asarray(root['Label'])

                samples_p = []
                samples_x = []
                samples_y = []
                for y in range(hwin, height - hwin):
                    for x in range(hwin, width - hwin):
                        label = None
                        if labels is not None:
                            label = labels[x, y]

                        bands = []
                        for f in features:
                            band = data_bands[f][(x - hwin):(x + hwin + 1), (y - hwin):(y + hwin + 1)]
                            bands.append(band)

                        sample = np.stack(bands)
                        # Move features from the first axis to the last axis.
                        sample = np.rollaxis(sample, 0, 3)

                        samples_p.append((x, y))
                        samples_x.append(sample)
                        samples_y.append(label)

                        if len(samples_x) == batch_size:
                            batch_p = np.stack(samples_p)
                            batch_x = np.stack(samples_x)
                            batch_y = np.stack(samples_y)

                            samples_p = []
                            samples_x = []
                            samples_y = []

                            yield batch_p, batch_x, batch_y
