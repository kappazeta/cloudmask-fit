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

import json

from cm_fit.model.architectures import ARCH_MAP
from cm_fit.util.json_codec import CMFJSONEncoder
from cm_fit.util import log as ulog


class CMConfig(ulog.Loggable):
    def __init__(self):
        super(CMConfig, self).__init__("CMF.cfg")

        self.cfg = {
            "version": 2,
            "input": {
                "path_dir": "input/"
            },
            "split": {
                "ratio": {
                    "test": 0.1,
                    "val": 0.2
                }
            },
            "model": {
                "architecture": "a1",
                "features": ["AOT", "B01", "B02", "B03", "B04", "B05", "B06", "B08", "B8A", "B09", "B11", "B12", "WVP"],
                "pixel_window_size": 9,
            },
            "train": {
                "learning_rate": 1E-4,
                "batch_size": 6,
                "num_epochs": 10
            },
            "predict": {
                "batch_size": 1
            }
        }

    def load(self, path):
        """
        Load configuration from a JSON file.
        :param path: Path to the JSON file.
        """
        self.path = path
        with open(path, "rt") as fi:
            self.cfg = json.load(fi)
            # TODO:: Validate config structure

    def get_dict(self):
        """
        Get configuration as dict.
        """
        return self.cfg

    def is_model_arch_pixel_based(self):
        """
        Check if the model architecture is pixel-based (True) or image-based (False).
        """
        name_arch = self.cfg["model"]["architecture"]
        if name_arch in ARCH_MAP.keys():
            return ARCH_MAP[name_arch].PIXEL_BASED
        return False
