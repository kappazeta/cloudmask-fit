import os
import numpy as np
import random


def generate_splits(path_input, val_ratio, test_ratio):
    all_indices = [f for f in os.listdir(path_input) if
                   os.path.isfile(os.path.join(path_input, f)) and os.path.join(path_input, f).endswith('.nc')]
    all_indices_fullname = [os.path.join(path_input, index) for index in all_indices]
    total_length = len(all_indices)

    validation_test_number = int((val_ratio + test_ratio) * total_length)
    test_number = int(test_ratio * total_length)

    val_test_indices = random.sample(all_indices_fullname, validation_test_number)
    test_indices = random.sample(val_test_indices, test_number)
    val_indices = list(set(val_test_indices) - set(test_indices))
    train_indices = list(set(all_indices_fullname) - set(val_test_indices))

    dictionary_out = {
        'total': total_length,
        'filepaths': all_indices_fullname,
        'test': test_indices,
        'val': val_indices,
        'train': train_indices
    }

    return dictionary_out
