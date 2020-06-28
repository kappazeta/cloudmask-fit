# vim: set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab

# Copyright 2020 KappaZeta Ltd.
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

import tensorflow as tf

from cm_fit.model.model import CMModel


class CMMA1(CMModel):
    def __init__(self):
        super(CMMA1, self).__init__("CMM.A1")

        self.pooling_size = (3, 3)

    def l_conv(self, filters, l_in, dropout=None):
        l = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(l_in)
        l = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(l)
        if dropout:
            l = tf.keras.layers.Dropout(dropout)(l)
        return l, tf.keras.layers.MaxPool2D(pool_size=self.pooling_size)(l)

    def l_up(self, filters, l_in, l_c):
        l = tf.keras.layers.UpSampling2D(self.pooling_size)(l_in)
        l = tf.keras.layers.Conv2D(filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(l)
        l = tf.keras.layers.concatenate([l_c, l], axis=3)
        l = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(l)
        l = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(l)

        return l

    def construct(self, width, height, num_channels, categories):
        # Based on the U-Net architecture, but down-scaled for single pixel samples with symmetrical neighbourhood.
        # For symmetrical neighbourhood, width and height must be odd numbers.
        self.input_shape = (width, height, num_channels)
        self.output_shape = (categories,)

        with tf.name_scope("Model"):
            l_in = tf.keras.layers.Input(self.input_shape, name='input')

            l_c1, l_p1 = self.l_conv(32, l_in)
            l_d2, l_p2 = self.l_conv(64, l_p1, dropout=0.5)

            l_c5 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(l_p2)
            l_c5 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(l_c5)
            l_d5 = tf.keras.layers.Dropout(0.5)(l_c5)

            l_c8 = self.l_up(64, l_d5, l_d2)
            l_c9 = self.l_up(32, l_c8, l_c1)

            l_c10 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(l_c9)
            l_c10 = tf.keras.layers.Conv2D(categories, 1, activation='sigmoid')(l_c10)
            l_out = tf.keras.layers.Flatten()(l_c10)
            l_out = tf.keras.layers.Dense(categories, activation='sigmoid', name='output')(l_out)

            self.model = tf.keras.Model(inputs=[l_in], outputs=[l_out])
