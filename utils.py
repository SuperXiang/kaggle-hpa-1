import argparse
import glob
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch import nn


def adjust_initial_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = lr


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def with_he_normal_weights(layer):
    nn.init.kaiming_normal_(layer.weight, a=0, mode="fan_in")
    return layer


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def kfold_split(n_splits, values, classes):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_value_indexes, test_value_indexes in skf.split(values, classes):
        train_values = [values[i] for i in train_value_indexes]
        test_values = [values[i] for i in test_value_indexes]
        yield train_values, test_values


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_lines(file_path):
    with open(file_path) as categories_file:
        return [l.rstrip("\n") for l in categories_file.readlines()]


def check_model_improved(old_score, new_score, threshold=1e-4):
    return new_score - old_score > threshold


def list_sorted_model_files(base_dir):
    return sorted(glob.glob("{}/model-*.pth".format(base_dir)), key=lambda e: int(os.path.basename(e)[6:-4]))


def calculate_balance_weights(df, target_df, num_classes):
    counts = np.zeros(num_classes)
    for target in df.Target:
        counts[np.asarray(target)] += 1

    median_count = np.median(counts)
    class_weights = np.asarray([median_count / c for c in counts])

    weights = [np.max(class_weights[np.asarray(target)]) for target in target_df.Target]

    return weights, class_weights.tolist()


def log(*args):
    print(*args, flush=True)


def log_args(args):
    log("Arguments:")
    for arg in vars(args):
        log("  {}: {}".format(arg, getattr(args, arg)))
    log()
