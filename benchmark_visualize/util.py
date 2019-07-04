import json
import re

import numpy as np
from scipy.stats import rankdata


def load(f):
    with open(f, "r") as file:
        return json.load(file)


def rename(i):
    name = i.replace("hyperboost-qrd", "HyperBoost")
    name = i.replace("hyperboost-ei", "HyperBoostEI")
    name = name.replace("smac", "SMAC")
    name = name.replace("random_2x", "Random x2")
    name = name.replace("_test", "")
    name = name.replace("_train", "")
    name = name.replace("_mean", "")
    name = name.replace("_std", "")
    return name


def mean_of_runs(data, keep_runs=None):
    result = {}
    for task, values in data.items():
        run = {}
        for method, trajectories in values.items():
            if keep_runs is not None and method not in keep_runs:
                continue
            loss_train = np.mean([i['loss_train'] for i in trajectories], axis=0)
            loss_test = np.mean([i['loss_test'] for i in trajectories], axis=0)
            run[method] = {
                "loss_train": loss_train,
                "loss_test": loss_test
            }
        result[task] = run

    return result


def multi_rank(*args):
    arrays = [np.array(arg) for arg in args]
    ranks = np.array([np.zeros_like(ar) for ar in arrays])
    shape = arrays[0].shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            values = [ar[i, j] for ar in arrays]
            ranks[:, i, j] = rankdata(values)
    return ranks


def dual_rank(a, b):
    a = np.array(a)
    b = np.array(b)
    a_res = np.zeros_like(a)
    b_res = np.zeros_like(b)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            val_a = a[i, j]
            val_b = b[i, j]
            a_res[i, j] = 1 if val_a < val_b else 1.5 if val_a == val_b else 2
            b_res[i, j] = 1 if val_b < val_a else 1.5 if val_a == val_b else 2
    return a_res, b_res


def rank_against(mean_runs, dual=False):
    methods = list(list(mean_runs.values())[0].keys())
    train = {}
    test = {}
    for method in methods:
        train[method] = [i[method]['loss_train'] for i in mean_runs.values()]
        test[method] = [i[method]['loss_test'] for i in mean_runs.values()]

    if dual:
        ranked_train = dual_rank(*train.values())
        ranked_test = dual_rank(*test.values())
    else:
        ranked_train = multi_rank(*train.values())
        ranked_test = multi_rank(*test.values())

    result = {}

    for index, method in enumerate(methods):
        result[f"{method}_mean_train"] = np.mean(ranked_train[index], axis=0)
        result[f"{method}_std_train"] = np.std(ranked_train[index], axis=0)
        result[f"{method}_mean_test"] = np.mean(ranked_test[index], axis=0)
        result[f"{method}_std_test"] = np.std(ranked_test[index], axis=0)

    return result


def mean_of_datasets(mean_runs):
    methods = list(list(mean_runs.values())[0].keys())
    result = {}

    for method in methods:
        method_train = [i[method]['loss_train'] for i in mean_runs.values()]
        method_test = [i[method]['loss_test'] for i in mean_runs.values()]
        result[f"{method}_mean_train"] = np.mean(method_train, axis=0)
        result[f"{method}_mean_test"] = np.mean(method_test, axis=0)
        result[f"{method}_std_train"] = np.std(method_train, axis=0)
        result[f"{method}_std_test"] = np.std(method_test, axis=0)

    return result


def matches(filename, name, prefix="results"):
    match = re.match(fr"{prefix}-{name}-[0-9]*\.json", filename)
    return False if match is None else True
