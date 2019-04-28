import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from benchmarks import config

# Settings
from benchmarks.combine_results import combine_all

dir = "."
in_progress = False
name = "Adaboost"


def matches(filename):
    match = re.match(fr"results-{name}-[0-9]*\.json", filename)
    return False if match is None else True


files = [filename for filename in os.listdir(dir) if matches(filename)]
print("matching files:", files)


def mean_of_runs(data):
    result = {}
    for task, values in data.items():
        run = {}
        for method, trajectories in values.items():
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


if __name__ == "__main__":
    r = combine_all({"dir": "LightQR", "name": "DecisionTree-stoch"},
                    {"dir": ".", "name": "DecisionTree-stoch"},
                    ["hyperboost-qrd"])
    r = {seed: mean_of_runs(i) for seed, i in r.items()}
    r = {seed: rank_against(i) for seed, i in r.items()}
    # r = {seed: mean_of_datasets(i) for seed, i in r.items()}
    results = r

    # results = {}
    # for file in files:
    #     seed = re.findall(r"[0-9]+", file)[0]
    #     with open(os.path.join(dir, file), "r") as f:
    #         data = json.load(f)
    #         r = mean_of_runs(data)
    #         r = rank_against(r, dual=False)
    #         # r = mean_of_datasets(r)
    #         results[seed] = r

    results = [results[str(i)] for i in config.SEEDS if str(i) in results.keys()]
    if in_progress:
        results = results[:-1]
    frames = [pd.DataFrame(i) for i in results]
    mean = pd.concat(frames).groupby(level=0).mean().iloc[1:]
    std = pd.concat(frames).groupby(level=0).std().iloc[1:]

    columns = [i for i in mean.columns if "mean_train" in i]
    # columns = ["smac_mean_train", "hyperboost-qrd_mean_train"]

    for i in columns:
        mean[i].plot()
        plt.fill_between(np.arange(mean[i].shape[0]), mean[i] - std[i], mean[i] + std[i], alpha=0.5)

    plt.legend()
    plt.show()
    print()
