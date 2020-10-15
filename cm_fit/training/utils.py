import numpy as np


def set_normalization(generator, split, sub_batch):
    samples = len(split) // sub_batch
    sum_std = []
    sum_mean = []
    for i in range(sub_batch):
        curr_std, curr_mean_list, curr_unique_list = generator.get_normal_par(
            split[i * samples:(i + 1) * samples])
        sum_std.append(curr_std)
        sum_mean.append(curr_mean_list)
    sum_std = np.asarray(sum_std)
    sum_mean = np.asarray(sum_mean)

    final_std = np.sum(sum_std, axis=0)
    final_std = final_std / sub_batch
    final_mean = np.sum(sum_mean, axis=0)
    final_mean = final_mean / sub_batch

    generator.set_std(final_std.tolist())
    generator.set_means(final_mean.tolist())
    return final_std.tolist(), final_mean.tolist()