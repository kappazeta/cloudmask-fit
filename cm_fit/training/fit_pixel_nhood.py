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

from .fit_base import CMFit
from cm_fit.data_generator.dg_pixel_nhood import DataGeneratorPixelNHood


class CMFitPixelNHood(CMFit):
    def __init__(self, png_mode=False):
        super(CMFitPixelNHood, self).__init__(png_mode=png_mode)

        self.model_name_prefix = "a1-{}".format(self.pixel_window_size)

        self.subtile_dim = self.dim

        # TODO:: Ideally would use KZPTS files to random-sample batches from multiple sub-tiles.
        # To avoid exceeding RAM without too much effort, skip some pixels from each sub-tile.
        self.subsample_factor = 0.001

        # Replace the dimensions of the input tensors with pixel window size.
        self.dim = self.pixel_window_size, self.pixel_window_size
        # For data generator, we'll still pass the original subtile dimensions as part of params.

    def create_data_generator(self, splits, params, png_form=False):
        return DataGeneratorPixelNHood(
            splits, **params, png_form=png_form, pixel_window_size=self.pixel_window_size,
            subsample_factor=self.subsample_factor
        )
