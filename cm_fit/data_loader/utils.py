import os
import numpy as np
import random


def generate_splits(path_input, val_ratio, test_products):
    """ Generate train and validation split accoring to ratio and tes split read from test products file"""
    all_indices = [f for f in os.listdir(path_input) if
                   os.path.isfile(os.path.join(path_input, f)) and os.path.join(path_input, f).endswith('.nc')]
    all_indices_fullname = [os.path.join(path_input, index) for index in all_indices]
    test_indices = []
    train_val_indices = []
    for filename in all_indices:
        filename_spl = filename.split("/")[-1]
        filename_sub = filename_spl.split("_")
        filename_sub = filename_sub[0] + "_" + filename_sub[1]
        if filename_sub in test_products:
            test_indices.append(os.path.join(path_input, filename))
        elif filename_sub not in test_products:
            train_val_indices.append(os.path.join(path_input, filename))
    total_length = len(train_val_indices)

    validation_test_number = int(val_ratio * total_length)

    val_test_indices = random.sample(train_val_indices, validation_test_number)
    val_indices = list(set(val_test_indices))
    train_indices = list(set(train_val_indices) - set(val_test_indices))

    dictionary_out = {
        'total': total_length,
        'filepaths': all_indices_fullname,
        'test': test_indices,
        'val': val_indices,
        'train': train_indices
    }

    return dictionary_out


def string_to_list(input_line):
    output_line = input_line.split(" [")[-1]
    output_line = output_line.replace("]", "")
    output_line = output_line.split(", ")
    output_line = [int(item) for item in output_line]
    return output_line
