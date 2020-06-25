#!/usr/bin/python3
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

import gc
import os
import json
import argparse
import numpy as np
from netCDF4 import Dataset

from cm_fit.util import log as ulog


class CMFit(ulog.Loggable):
    def __init__(self):
        super(CMFit, self).__init__("CMF")

        self.cfg = {
            "input": {
                "path_dir": "input/"
            }
        }

    def load_config(self, path):
        with open(path, "rt") as fi:
            self.cfg = json.load(fi)

            # TODO:: Validate config structure

    def process_input(self):
        path_input_dir = self.cfg["input"]["path_dir"]
        for fname in os.listdir(path_input_dir):
            path_input_file = os.path.join(path_input_dir, fname)
            if os.path.isfile(path_input_file) and fname.endswith(".nc"):
                print(path_input_file)
                with Dataset(path_input_file, "r") as root:
                    print(root["AOT"].shape, np.asarray(root["AOT"]))



def main():
    # Parse command-line arguments.
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("-c", "--config", action="store", dest="path_config", default="config/config_example.json",
                   help="Path to the configuration file.")
    p.add_argument("-l", "--log", action="store", dest="logfile_path", default=None,
                   help="Path for a log file, if desired.")
    p.add_argument("-v", "--verbosity", dest="verbosity", type=int, action="store", default=3,
                   help="Level of verbosity (1 - 3).")

    args = p.parse_args()

    log = None
    try:
        # Initialize logging.
        log = ulog.init_logging(
            args.verbosity, "CloudMask Fit", "CMF",
            logfile=args.logfile_path
        )

        cmf = CMFit()
        cmf.load_config(args.path_config)
        cmf.process_input()

    except Exception as e:
        if log is not None:
            log.exception("Unhandled exception")
        else:
            print("Failed to initialize error logging")
            raise e


if __name__ == "__main__":
    main()

gc.collect()
