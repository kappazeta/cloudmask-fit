
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# https://github.com/virginie-do/DeepUNet/blob/master/model.py
"""
Created on Mon Jun 11 09:40:42 2018
@author: Virginie Do
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from cm_fit.model.model import CMModel
from keras.regularizers import l2


class NestedUnet(CMModel):
    def __init__(self):
        print('build DeepUnet ...')
        super(NestedUnet, self).__init__("Unet")

    def standard_unit(self, input_tensor, stage, nb_filter, kernel_size=3):

        act = 'elu'

        x = tf.keras.layers.Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv' + stage + '_1',
                   kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
        x = tf.keras.layers.Dropout(self.dropout_rate, name='dp' + stage + '_1')(x)
        x = tf.keras.layers.Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv' + stage + '_2',
                   kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
        x = tf.keras.layers.Dropout(self.dropout_rate, name='dp' + stage + '_2')(x)

        return x

    def construct(self, width, height, num_channels, num_categories, pretrained_weights=False, deep_supervision=False):

        # Handle Dimension Ordering for different backends
        global bn_axis
        self.input_shape = (width, height, num_channels)
        self.output_shape = (num_categories,)
        self.dropout_rate = 0.5
        self.nb_filter = [64, 128, 256, 512, 1024]

        with tf.name_scope("Model"):
            img_input = tf.keras.layers.Input(shape=self.input_shape, name='main_input')

            conv1_1 = self.standard_unit(img_input, stage='11', nb_filter=self.nb_filter[0])
            pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

            conv2_1 = self.standard_unit(pool1, stage='21', nb_filter=self.nb_filter[1])
            pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

            up1_2 = tf.keras.layers.Conv2DTranspose(self.nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
            conv1_2 = tf.keras.layers.concatenate([up1_2, conv1_1], name='merge12', axis=3)
            conv1_2 = self.standard_unit(conv1_2, stage='12', nb_filter=self.nb_filter[0])

            conv3_1 = self.standard_unit(pool2, stage='31', nb_filter=self.nb_filter[2])
            pool3 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

            up2_2 = tf.keras.layers.Conv2DTranspose(self.nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
            conv2_2 = tf.keras.layers.concatenate([up2_2, conv2_1], name='merge22', axis=3)
            conv2_2 = self.standard_unit(conv2_2, stage='22', nb_filter=self.nb_filter[1])

            up1_3 = tf.keras.layers.Conv2DTranspose(self.nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
            conv1_3 = tf.keras.layers.concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=3)
            conv1_3 = self.standard_unit(conv1_3, stage='13', nb_filter=self.nb_filter[0])

            conv4_1 = self.standard_unit(pool3, stage='41', nb_filter=self.nb_filter[3])
            pool4 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

            up3_2 = tf.keras.layers.Conv2DTranspose(self.nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
            conv3_2 = tf.keras.layers.concatenate([up3_2, conv3_1], name='merge32', axis=3)
            conv3_2 = self.standard_unit(conv3_2, stage='32', nb_filter=self.nb_filter[2])

            up2_3 = tf.keras.layers.Conv2DTranspose(self.nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
            conv2_3 = tf.keras.layers.concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=3)
            conv2_3 = self.standard_unit(conv2_3, stage='23', nb_filter=self.nb_filter[1])

            up1_4 = tf.keras.layers.Conv2DTranspose(self.nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
            conv1_4 = tf.keras.layers.concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=3)
            conv1_4 = self.standard_unit(conv1_4, stage='14', nb_filter=self.nb_filter[0])

            conv5_1 = self.standard_unit(pool4, stage='51', nb_filter=self.nb_filter[4])

            up4_2 = tf.keras.layers.Conv2DTranspose(self.nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
            conv4_2 = tf.keras.layers.concatenate([up4_2, conv4_1], name='merge42', axis=3)
            conv4_2 = self.standard_unit(conv4_2, stage='42', nb_filter=self.nb_filter[3])

            up3_3 = tf.keras.layers.Conv2DTranspose(self.nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
            conv3_3 = tf.keras.layers.concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=3)
            conv3_3 = self.standard_unit(conv3_3, stage='33', nb_filter=self.nb_filter[2])

            up2_4 = tf.keras.layers.Conv2DTranspose(self.nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
            conv2_4 = tf.keras.layers.concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=3)
            conv2_4 = self.standard_unit(conv2_4, stage='24', nb_filter=self.nb_filter[1])

            up1_5 = tf.keras.layers.Conv2DTranspose(self.nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
            conv1_5 = tf.keras.layers.concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=3)
            conv1_5 = self.standard_unit(conv1_5, stage='15', nb_filter=self.nb_filter[0])

            nestnet_output_1 = tf.keras.layers.Conv2D(num_categories, (1, 1), activation='sigmoid', name='output_1',
                                      kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
            nestnet_output_2 = tf.keras.layers.Conv2D(num_categories, (1, 1), activation='sigmoid', name='output_2',
                                      kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
            nestnet_output_3 = tf.keras.layers.Conv2D(num_categories, (1, 1), activation='sigmoid', name='output_3',
                                      kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
            nestnet_output_4 = tf.keras.layers.Conv2D(num_categories, (1, 1), activation='sigmoid', name='output_4',
                                      kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

            if deep_supervision:
                self.model = tf.keras.Model(input=img_input, output=[nestnet_output_1,
                                                       nestnet_output_2,
                                                       nestnet_output_3,
                                                       nestnet_output_4])
            else:
                self.model = tf.keras.Model(img_input, nestnet_output_4)

        return self.model


