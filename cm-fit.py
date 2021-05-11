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
from cm_fit.training.cm_initialize import CMFit


def main():
    # Parse command-line arguments.
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("-train", "--train", action="store", dest="train", default="unet",
                   help="Training mode, argument pass a name for saved weights")
    p.add_argument("-pretrain", "--pretrain", action="store", dest="pretrain", default="",
                   help="Pretrained weights, expect absolute path for pretrained .hdf5 weight file")
    p.add_argument("-predict", "--predict", action="store", dest="predict", default=None,
                   help="Prediction mode.")
    p.add_argument("-tune", "--tune", action="store", dest="tune", default=None,
                   help="Parameter tuning mode, argument pass a name for saved weights")
    p.add_argument("-w", "--weights", action="store", dest="weights", default=None,
                   help="Path to the model weights to use for prediction.")
    p.add_argument("-c", "--config", action="store", dest="path_config", default="config/config_example.json",
                   help="Path to the configuration file.")
    p.add_argument("-l", "--log", action="store", dest="logfile_path", default=None,
                   help="Path for a log file, if desired.")
    p.add_argument("-v", "--verbosity", dest="verbosity", type=int, action="store", default=3,
                   help="Level of verbosity (1 - 3).")
    p.add_argument("-d", "--dev_mode", dest="dev_mode", default=False,
                   help="Using other data_generator")
    p.add_argument("-val", "--validate", dest="validate", action="store", default=None,
                   help="Validation running")
    p.add_argument("-test", "--test", dest="test", action="store", default=None,
                   help="Testing for product name")
    p.add_argument("-aug", "--augment", dest="augmentation", action="store", default=None,
                   help="Allow data augmentation")
    p.add_argument("-stat", "--stats", dest="statistic", action="store", default=None,
                   help="Show label distribution")
    p.add_argument("-png", "--train_png", dest="train_png", action="store", default=None,
                   help="training for 3 features")
    p.add_argument("-select", "--selecting", dest="selecting", action="store", default=None,
                   help="training for 3 features")
    p.add_argument("-orig", "--original_rgb", dest="original_rgb", action="store", default=None,
                   help="save original rgb")

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
        # Read test products list and put it separately
        parsed_test_products = cmf.load_test_products()
        print(parsed_test_products)

        if args.selecting:
            """ Mode for running sub-tiles prediction on all files that consist in folder """
            cmf.split()
            cmf.selecting(args.selecting, args.weights)
        elif args.original_rgb:
            """ Mode for storing original images and comparison masks without running prediction """
            cmf.split()
            cmf.get_origin_im()
        elif args.train_png:
            """ Mode for training model only on RGB bands (segments-ai) """
            cmf.split()
            cmf.train(args.train)
        elif args.predict is not None:
            """ Mode for prediction on validation set that is generated in split file,
                files should have labels for confusion matrix and metrics calculation"""
            cmf.predict(args.predict, args.weights, parsed_test_products)
        elif args.validate:
            """ Mode for prediction on all files in specific folder that should have labels, 
                calculation metrics and confusion matrix """
            cmf.validation(args.validate, args.weights, parsed_test_products)
        elif args.train:
            """ Mode for training model """
            cmf.split()
            if args.pretrain:
                cmf.train(parsed_test_products, args.train, args.pretrain)
            else:
                cmf.train(parsed_test_products, args.train)
        elif args.tune:
            """ Parameter tuning mode """
            cmf.split()
            cmf.parameter_tune(parsed_test_products, args.tune)
        elif args.test:
            """ Evaluation on all test dataset that marked in parsed_test_products """
            cmf.test(args.test, args.weights, parsed_test_products)
        elif args.statistic:
            """ Output per class statistic for labels """
            cmf.run_stats()
        
    except Exception as e:
        if log is not None:
            log.exception("Unhandled exception")
        else:
            print("Failed to initialize error logging")
            raise e


if __name__ == "__main__":
    main()

gc.collect()
