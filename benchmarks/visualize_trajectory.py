import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from benchmarks import config

# Settings
dir = "LightEPM_QR"
in_progress = False
name = "DecisionTree"

files = [filename for filename in os.listdir(dir) if filename.startswith(f"results-{name}")]


def mean_of_runs(data):
    result = {}
    for task, values in data.items():
        run = {}
        for method, trajectories in values.items():
            loss_train = np.mean([i['loss_train'] for i in trajectories], axis=0)
            loss_test = np.mean([i['loss_train'] for i in trajectories], axis=0)
            run[method] = {
                "loss_train": loss_train,
                "loss_test": loss_test
            }
        result[task] = run

    return result

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


def rank_against(mean_runs):
    smac_train = [i['smac']['loss_train'] for i in mean_runs.values()]
    hyperboost_train = [i['hyperboost']['loss_train'] for i in mean_runs.values()]
    smac_test = [i['smac']['loss_test'] for i in mean_runs.values()]
    hyperboost_test = [i['hyperboost']['loss_test'] for i in mean_runs.values()]

    smac_train, hyperboost_train = dual_rank(smac_train, hyperboost_train)
    smac_test, hyperboost_test = dual_rank(smac_test, hyperboost_test)

    return {
        "smac_train_mean": np.mean(smac_train, axis=0),
        "smac_train_std": np.std(smac_train, axis=0),
        "smac_test_mean": np.mean(smac_test, axis=0),
        "smac_test_std": np.std(smac_test, axis=0),
        "hyperboost_train_mean": np.mean(hyperboost_train, axis=0),
        "hyperboost_train_std": np.std(hyperboost_train, axis=0),
        "hyperboost_test_mean": np.mean(hyperboost_test, axis=0),
        "hyperboost_test_std": np.std(hyperboost_test, axis=0),
    }

def mean_of_datasets(mean_runs):
    smac_train = [i['smac']['loss_train'] for i in mean_runs.values()]
    hyperboost_train = [i['hyperboost']['loss_train'] for i in mean_runs.values()]
    smac_test = [i['smac']['loss_test'] for i in mean_runs.values()]
    hyperboost_test = [i['hyperboost']['loss_test'] for i in mean_runs.values()]

    return {
        "smac_train_mean": np.mean(smac_train, axis=0),
        "smac_train_std": np.std(smac_train, axis=0),
        "smac_test_mean": np.mean(smac_test, axis=0),
        "smac_test_std": np.std(smac_test, axis=0),
        "hyperboost_train_mean": np.mean(hyperboost_train, axis=0),
        "hyperboost_train_std": np.std(hyperboost_train, axis=0),
        "hyperboost_test_mean": np.mean(hyperboost_test, axis=0),
        "hyperboost_test_std": np.std(hyperboost_test, axis=0),
    }

if __name__ == "__main__":
    results = {}
    for file in files:
        seed = re.findall(r"[0-9]+", file)[0]
        with open(os.path.join(dir, file), "r") as f:
            data = json.load(f)
            r = mean_of_runs(data)
            r = rank_against(r)
            # r = mean_of_datasets(r)
            results[seed] = r

    results = [results[str(i)] for i in config.SEEDS if str(i) in results.keys()]
    if in_progress:
        results = results[:-1]
    frames = [pd.DataFrame(i) for i in results]
    mean = pd.concat(frames).groupby(level=0).mean().iloc[1:]
    std = pd.concat(frames).groupby(level=0).std().iloc[1:]
    mean["smac_train_mean"].plot()
    mean["hyperboost_train_mean"].plot()
    plt.fill_between(np.arange(mean["smac_train_mean"].shape[0]), mean["smac_train_mean"] - std["smac_train_mean"], mean["smac_train_mean"] + std["smac_train_mean"], alpha=0.5)
    plt.fill_between(np.arange(mean["hyperboost_train_mean"].shape[0]), mean["hyperboost_train_mean"] - std["hyperboost_train_mean"],
                     mean["hyperboost_train_mean"] + std["hyperboost_train_mean"], alpha=0.5)
    plt.legend()
    plt.show()
    print()