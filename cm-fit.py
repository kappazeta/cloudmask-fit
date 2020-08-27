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
import argparse

from cm_fit.util import log as ulog
from cm_fit.cm_fit import CMFit
from cm_fit.training.cm_initialize import CMInit


def main():
    # Parse command-line arguments.
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("-t", "--train", action="store_true", dest="train", default=True,
                   help="Train a new model.")
    p.add_argument("-p", "--predict", action="store", dest="predict", default=None,
                   help="Path to predict on.")
    p.add_argument("-w", "--weights", action="store", dest="weights", default=None,
                   help="Path to the model weights to use for prediction.")
    p.add_argument("-c", "--config", action="store", dest="path_config", default="config/config_example.json",
                   help="Path to the configuration file.")
    p.add_argument("-l", "--log", action="store", dest="logfile_path", default=None,
                   help="Path for a log file, if desired.")
    p.add_argument("-v", "--verbosity", dest="verbosity", type=int, action="store", default=3,
                   help="Level of verbosity (1 - 3).")
    p.add_argument("-dev", "--dev_mode", dest="dev_mode", default=False,
                   help="Using other data_generator")

    args = p.parse_args()

    log = None
    try:
        # Initialize logging.
        log = ulog.init_logging(
            args.verbosity, "CloudMask Fit", "CMF",
            logfile=args.logfile_path
        )

        if args.dev_mode:
            cmf = CMInit()
            cmf.load_config(args.path_config)
            print("Development mode")
            cmf.split()
            # Generators
            cmf.train()
        else:
            cmf = CMFit()
            cmf.load_config(args.path_config)
            if args.predict is not None:
                cmf.predict(args.predict, args.weights)
            elif args.train:
                cmf.split()
                cmf.train()

    except Exception as e:
        if log is not None:
            log.exception("Unhandled exception")
        else:
            print("Failed to initialize error logging")
            raise e


if __name__ == "__main__":
    main()

gc.collect()
