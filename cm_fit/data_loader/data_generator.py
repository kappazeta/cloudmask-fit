import numpy as np
import netCDF4 as nc
import os
import random
import keras
import skimage.io as skio
from keras.utils import np_utils
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence
from cm_fit.data_loader.data_utils import string_to_list


class DataGenerator(Sequence):
    def __init__(self, list_indices, path_input, batch_size, features, dim, num_classes, label_set, normalization,
                 product_level, shuffle=True, png_form=False):
        """ Initialization """
        self.path = path_input
        if product_level == "L2A":
            self.stds = [0.00085, 0.041, 0.039, 0.036, 0.035, 0.036, 0.035, 0.034, 0.035, 0.034, 0.06, 0.024, 0.0211,
                         0.012]
            self.means = [0.0016, 0.029, 0.028, 0.028, 0.027, 0.033, 0.0445, 0.0482, 0.05, 0.05, 0.069, 0.032, 0.024,
                          0.02]
            self.min_v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.max_v = [0.0042, 0.298, 0.282, 0.267, 0.256, 0.253, 0.246, 0.245, 0.245, 0.242, 0.250, 0.233, 0.235,
                          0.074]
        elif product_level == "L1C":
            self.stds = [0.0246, 0.025, 0.0251, 0.0278, 0.0277, 0.0284, 0.030, 0.0292, 0.0309, 0.0168, 0.0045, 0.0213,
                         0.0175]
            self.means = [0.0321, 0.0284, 0.0259, 0.0246, 0.0282, 0.0400, 0.0453, 0.0439, 0.0484, 0.0188, 0.0024,
                          0.0288, 0.0197]
            self.min_v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.max_v = [0.21, 0.326, 0.264, 0.298, 0.245, 0.258, 0.266, 0.364, 0.259, 0.165, 0.055, 0.203, 0.208]
        self.list_indices = list_indices
        self.total_length = len(self.list_indices)
        self.batch_size = batch_size
        self.normalization = normalization
        if png_form:
            self.features = ["TCI_R", "TCI_G", "TCI_B"]
        else:
            self.features = features
        self.dim = dim
        self.label_set = label_set
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.png_form = png_form
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

        return X, y#, sample_weights

    def set_std(self, stds):
        self.stds = stds

    def set_means(self, means):
        self.means = means

    def set_min(self, min_v):
        self.min_v = min_v

    def set_max(self, max_v):
        self.max_v = max_v

    def set_normalization_file(self, meta_data_path):
        """ Read file with std, mean, min and max values and overwrite coefficients """
        if os.path.isfile(os.path.join(meta_data_path, "normalization.txt")):
            with open(os.path.join(meta_data_path, "normalization.txt")) as file:
                all_lines = file.readlines()
                self.stds = string_to_list(all_lines[0])
                self.means = string_to_list(all_lines[1])
                self.min_v = string_to_list(all_lines[2])
                self.max_v = string_to_list(all_lines[3])

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_indices))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_label_stat(self, list_indices_temp, classes):
        overall_stat = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        per_image_stat = []
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith('.nc'):
                file = file.replace('merged_l1c_1507', 'kappamask_maja_att2')
                with nc.Dataset(file, 'r') as root:
                    try:
                        label = np.asarray(root[self.label_set])
                        unique_elements, counts_elements = np.unique(label, return_counts=True)
                        curr_dic = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
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

        if sum(overall_stat.values()) == 0:
            return overall_stat, per_image_stat
        else:
            overall_stat = {k: v / sum(overall_stat.values()) for k, v in overall_stat.items()}
            return overall_stat, per_image_stat

    def store_orig(self, list_indices_temp, path_prediction):
        """Save labels to folder"""
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith('.nc'):
                file = file.replace('merged_l1c_1507', 'kappamask_maja_att2')
                with nc.Dataset(file, 'r') as root:
                    data_bands = [np.asarray(root[f])
                                  for i, f in enumerate(["TCI_R", "TCI_G", "TCI_B"])]

                    data_bands = np.stack(data_bands)
                    data_bands = np.rollaxis(data_bands, 0, 3)

                    file_name = file.split(".")[0].split("/")[-1]

                    if not os.path.exists(path_prediction + "/" + file_name):
                        os.mkdir(path_prediction + "/" + file_name)

                    try:
                        sen2cor_cc = np.asarray(root['S2CC'])
                        sen2cor_cs = np.asarray(root['S2CS'])
                        sen2cor_scl = np.asarray(root['SCL'])
                        sen2cor_scl = sen2cor_scl * 63 + 3
                        sen2cor_scl[sen2cor_scl > 255] = 20

                        sen2cor_scl = sen2cor_scl.astype(np.uint8)
                        # skio.imsave(saving_path + "/" + filename_image + "/prediction.png", classification)
                        im = Image.fromarray(sen2cor_scl)
                        im.save(path_prediction + "/" + file_name + "/SCL.png")
                    except:
                        print("Sen2Cor not found")

                    try:
                        fmask = np.asarray(root['FMC'])
                        fmask = fmask * 63 + 3
                        fmask[fmask > 255] = 20
                        fmask = fmask.astype(np.uint8)
                        # skio.imsave(saving_path + "/" + filename_image + "/prediction.png", classification)
                        im = Image.fromarray(fmask)
                        im.save(path_prediction + "/" + file_name + "/FMC.png")
                    except:
                        print("FMASK not found")

                    try:
                        s2cloudls = np.asarray(root['SS2C'])
                        s2cloudl_cloud = np.asarray(root['SS2CC'])
                        s2cloudls = s2cloudls * 63 + 3
                        s2cloudls[s2cloudls > 255] = 20

                        s2cloudls = s2cloudls.astype(np.uint8)
                        # skio.imsave(saving_path + "/" + filename_image + "/prediction.png", classification)
                        im = Image.fromarray(s2cloudls)
                        im.save(path_prediction + "/" + file_name + "/SS2C_sinergise.png")

                        s2cloudl_cloud = s2cloudl_cloud * 255
                        s2cloudl_cloud = s2cloudl_cloud.astype(np.uint8)
                        # skio.imsave(saving_path + "/" + filename_image + "/prediction.png", classification)
                        im = Image.fromarray(s2cloudl_cloud)
                        im.save(path_prediction + "/" + file_name + "/SS2CC_sinergise_cloudonly.png")
                    except:
                        print("sinergise not found")
                    try:
                        label = np.asarray(root[self.label_set])
                        y = np_utils.to_categorical(label, self.num_classes)
                        label = label * 63 + 3
                        label[label > 255] = 20
                        label = label.astype(np.uint8)
                        # skio.imsave(saving_path + "/" + filename_image + "/prediction.png", classification)
                        im = Image.fromarray(label)
                        im.save(path_prediction + "/" + file_name + "/label.png")
                    except:
                        print("Label not found")
                    # Lossy conversion Range [-0.5882352590560913, 6.766853332519531].
                    unique_before = np.unique(data_bands)
                    # data_bands *= 255
                    data_bands = data_bands.astype(np.uint8)
                    skio.imsave(path_prediction + "/" + file_name + "/orig.png", data_bands)

                    # sen2cor_cc = sen2cor_cc.astype(np.uint8)
                    # sen2cor_cs = sen2cor_cs.astype(np.uint8)
                    # sen2cor_cs *= 255
                    #sen2cor_scl = sen2cor_scl * 63 + 3

                    #sen2cor_scl = sen2cor_scl.astype(np.uint8)
                    # skio.imsave(saving_path + "/" + filename_image + "/prediction.png", classification)
                    #im = Image.fromarray(sen2cor_scl)
                    #im.save(path_prediction + "/" + file_name + "/SCL.png")

                    #skio.imsave(path_prediction + "/" + file_name + "/S2CC.png", sen2cor_cc)

    def get_labels(self, list_indices_temp, path_prediction, classes):
        """Save labels to folder"""
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith('.nc'):
                file = file.replace('merged_l1c_1507', 'kappamask_maja_att2')
                with nc.Dataset(file, 'r') as root:
                    y = np.zeros((self.dim[0], self.dim[1], self.num_classes), dtype=np.float32)
                    data_bands = [np.asarray(root[f])
                                  for i, f in enumerate(["TCI_R", "TCI_G", "TCI_B"])]

                    data_bands = np.stack(data_bands)
                    data_bands = np.rollaxis(data_bands, 0, 3)
                    try:
                        label = np.asarray(root[self.label_set])
                        y = np_utils.to_categorical(label, self.num_classes)
                    except:
                        print("No label for " + file)
                    # img = Image.fromarray(data_bands, 'RGB')
                    file_name = file.split(".")[0].split("/")[-1]
                    # img.save(path_prediction+"/"+file_name+"orig.png")

                    if not os.path.exists(path_prediction + "/" + file_name):
                        os.mkdir(path_prediction + "/" + file_name)

                    # Lossy conversion Range [-0.5882352590560913, 6.766853332519531].
                    unique_before = np.unique(data_bands)
                    # data_bands *= 255
                    data_bands = data_bands.astype(np.uint8)
                    skio.imsave(path_prediction + "/" + file_name + "/orig.png", data_bands)

                    label = label * 63 + 3
                    label[label > 255] = 20
                    label = label.astype(np.uint8)
                    # skio.imsave(saving_path + "/" + filename_image + "/prediction.png", classification)
                    im = Image.fromarray(label)
                    im.save(path_prediction + "/" + file_name + "/label.png")

                    """for j, curr_cl in enumerate(classes):
                        saving_filename = path_prediction + "/" + file_name + "/" + curr_cl
                        curr_array = y[:, :, j].copy()
                        curr_array *= 255
                        curr_array = curr_array.astype(np.uint8)
                        skio.imsave(saving_filename + ".png", curr_array)"""
        return True

    def get_normal_par(self, list_indices_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        X = np.zeros((len(list_indices_temp), self.dim[0], self.dim[1], len(self.features)), dtype=np.float32)
        # Initialization
        for i, file in enumerate(list_indices_temp):
            file = file.replace('merged_l1c_1507', 'kappamask_maja_att2')
            if os.path.isfile(file) and file.endswith('.nc'):
                with nc.Dataset(file, 'r') as root:
                    data_bands = [np.asarray(root[f]) for f in self.features]
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
        X = np.zeros((self.batch_size, self.dim[0], self.dim[1], len(self.features)), dtype=np.float32)
        y = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.num_classes), dtype=np.float32)
        # Initialization
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith('.nc'):
                file = file.replace('merged_l1c_1507', 'kappamask_maja_att2')
                filename = file.split("/")[-1]
                filename_sub = filename.split("_")
                filename_sub = filename_sub[0] + "_" + filename_sub[1]
                with nc.Dataset(file, 'r') as root:
                    if self.png_form:
                        data_bands = [(np.asarray(root[f]))*1/255 for i, f in
                                      enumerate(self.features)]
                    else:
                        if self.normalization == "minmax":
                            data_bands = [(np.asarray(root[f]) - self.min_v[i]) / (self.max_v[i]-self.min_v[i]) for i, f in
                                          enumerate(self.features)]
                        else:
                            data_bands = [(np.asarray(root[f]) - self.means[i]) / (self.stds[i]) for i, f
                                          in enumerate(self.features)]
                    try:
                        label = np.asarray(root[self.label_set])
                        unique_lbl = np.unique(label)

                        y[i] = np_utils.to_categorical(label, self.num_classes)
                    except:
                        print("Label for " + file + " not found")
                        print(data_bands[0].shape)
                    data_bands = np.stack(data_bands)
                    data_bands = np.rollaxis(data_bands, 0, 3)
                    # data_bands = data_bands.reshape((self.dim[0], self.dim[1], len(self.features)))
                    X[i,] = data_bands

        return X, y#, sample_weigths

    def get_classes(self):
        y = np.zeros((len(self.list_indices), self.dim[0], self.dim[1], self.num_classes), dtype=np.float32)
        # Initialization
        for i, file in enumerate(self.list_indices):
            if os.path.isfile(file) and file.endswith('.nc'):
                file = file.replace('merged_l1c_1507', 'kappamask_maja_att2')
                with nc.Dataset(file, 'r') as root:
                    try:
                        label = np.asarray(root[self.label_set])
                        y[i] = np_utils.to_categorical(label, self.num_classes)
                    except:
                        print("Label for " + file + " not found")
        return y

    def get_sen2cor(self, store_path):
        y = np.zeros((len(self.list_indices), self.dim[0], self.dim[1], self.num_classes), dtype=np.float32)
        # Initialization
        for i, file in enumerate(self.list_indices):
            if os.path.isfile(file) and file.endswith('.nc'):
                file = file.replace('merged_l1c_1507', 'kappamask_maja_att2')
                with nc.Dataset(file, 'r') as root:
                    try:
                        sen2cor = np.asarray(root["SCL"])
                        y[i] = np_utils.to_categorical(sen2cor, self.num_classes)

                        sen2cor = sen2cor * 63 + 3
                        sen2cor[sen2cor > 255] = 20

                        sen2cor = sen2cor.astype(np.uint8)
                        # skio.imsave(saving_path + "/" + filename_image + "/prediction.png", classification)
                        im = Image.fromarray(sen2cor)
                        file_name = file.split(".")[0].split("/")[-1]
                        if not os.path.exists(store_path + "/" + file_name):
                            os.mkdir(store_path + "/" + file_name)
                        im.save(store_path + "/" + file_name + "/SCL.png")
                    except:
                        print("Sen2Cor for confusion " + file + " not found")
        return y

    def get_s2cloudless(self, store_path):
        y = np.zeros((len(self.list_indices), self.dim[0], self.dim[1], self.num_classes), dtype=np.float32)
        # Initialization
        for i, file in enumerate(self.list_indices):
            if os.path.isfile(file) and file.endswith('.nc'):
                file = file.replace('merged_l1c_1507', 'kappamask_maja_att2')
                with nc.Dataset(file, 'r') as root:
                    try:
                        ss2c = np.asarray(root["SS2C"])
                        y[i] = np_utils.to_categorical(ss2c, self.num_classes)

                        ss2c = ss2c * 63 + 3
                        ss2c[ss2c > 255] = 20

                        ss2c = ss2c.astype(np.uint8)
                        # skio.imsave(saving_path + "/" + filename_image + "/prediction.png", classification)
                        im = Image.fromarray(ss2c)
                        file_name = file.split(".")[0].split("/")[-1]
                        im.save(store_path + "/" + file_name + "/SS2C.png")
                    except:
                        print("S2cloudless for confusion " + file + " not found")
        return y

    def get_maja(self, store_path):
        y = np.zeros((len(self.list_indices), self.dim[0], self.dim[1], self.num_classes), dtype=np.float32)
        # Initialization
        for i, file in enumerate(self.list_indices):
            if os.path.isfile(file) and file.endswith('.nc'):
                file = file.replace('merged_l1c_1507', 'kappamask_maja_att2')
                with nc.Dataset(file, 'r') as root:
                    try:
                        maja = np.asarray(root["MAJAC"])
                        y[i] = np_utils.to_categorical(maja, self.num_classes)

                        maja = maja * 63 + 3
                        maja[maja > 255] = 20

                        maja = maja.astype(np.uint8)
                        # skio.imsave(saving_path + "/" + filename_image + "/prediction.png", classification)
                        im = Image.fromarray(maja)
                        file_name = file.split(".")[0].split("/")[-1]
                        im.save(store_path + "/" + file_name + "/MAJA.png")
                    except:
                        print("Maja for confusion " + file + " not found")
        return y

    def get_fmask(self, store_path):
        y = np.zeros((len(self.list_indices), self.dim[0], self.dim[1], self.num_classes), dtype=np.float32)
        # Initialization
        for i, file in enumerate(self.list_indices):
            if os.path.isfile(file) and file.endswith('.nc'):
                file = file.replace('merged_l1c_1507', 'kappamask_maja_att2')
                with nc.Dataset(file, 'r') as root:
                    try:
                        fmask = np.asarray(root["FMC"])
                        y[i] = np_utils.to_categorical(fmask, self.num_classes)

                        fmask = fmask * 63 + 3
                        fmask[fmask > 255] = 20

                        fmask = fmask.astype(np.uint8)
                        # skio.imsave(saving_path + "/" + filename_image + "/prediction.png", classification)
                        im = Image.fromarray(fmask)
                        file_name = file.split(".")[0].split("/")[-1]
                        im.save(store_path + "/" + file_name + "/FMASK.png")
                    except:
                        print("Fmask for confusion " + file + " not found")
        return y

    def get_dl_l8s2(self, store_path):
        y = np.zeros((len(self.list_indices), self.dim[0], self.dim[1], self.num_classes), dtype=np.float32)
        # Initialization
        for i, file in enumerate(self.list_indices):
            if os.path.isfile(file) and file.endswith('.nc'):
                # TODO: the replacement coming because the path is different in split configuration file.
                #  Either create a mode to read from other path or reconfigure splitting file
                file = file.replace('merged_l1c_1507', 'kappamask_test_dl')
                with nc.Dataset(file, 'r') as root:
                    try:
                        dl_l8s2 = np.asarray(root["DL-L8S2-UV"])
                        y[i] = np_utils.to_categorical(dl_l8s2, self.num_classes)

                        dl_l8s2 = dl_l8s2 * 63 + 3
                        dl_l8s2[dl_l8s2 > 255] = 20

                        dl_l8s2 = dl_l8s2.astype(np.uint8)
                        # skio.imsave(saving_path + "/" + filename_image + "/prediction.png", classification)
                        im = Image.fromarray(dl_l8s2)
                        file_name = file.split(".")[0].split("/")[-1]
                        im.save(store_path + "/" + file_name + "/dl_l8s2.png")
                    except:
                        print("dl_l8s2 for confusion " + file + " not found")
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
        X = np.zeros((self.batch_size, self.dim[0], self.dim[1], len(self.features)), dtype=np.float32)
        y = np.zeros((self.batch_size, self.dim[0], self.dim[1]), dtype=np.float32)
        # Initialization
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith('.nc'):
                with nc.Dataset(file, 'r') as root:
                    data_bands = [np.asarray(root[f]) for f in self.features]
                    try:
                        label = np.asarray(root[self.label_set])
                        y[i] = label
                        # data_bands = np.stack(data_bands)
                        # data_bands = data_bands.reshape((self.dim[0], self.dim[1], len(self.features)))
                        # X[i,] = data_bands
                    except:
                        print("Label for " + file + " not found")
                        #print(data_bands[0].shape)
                        # y[i] = np.zeros_like(data_bands[0])
                    data_bands = np.stack(data_bands)
                    data_bands = data_bands.reshape((self.dim[0], self.dim[1], len(self.features)))
                    X[i,] = data_bands
        return X, y
