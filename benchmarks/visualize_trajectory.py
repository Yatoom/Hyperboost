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


def mean_of_datasets(mean_of_runs):
    smac_train = [i['smac']['loss_train'] for i in mean_of_runs.values()]
    hyperboost_train = [i['hyperboost']['loss_train'] for i in mean_of_runs.values()]
    smac_test = [i['smac']['loss_test'] for i in mean_of_runs.values()]
    hyperboost_test = [i['hyperboost']['loss_test'] for i in mean_of_runs.values()]

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
            r = mean_of_datasets(r)
            results[seed] = r

    results = [results[str(i)] for i in config.SEEDS if str(i) in results.keys()]
    if in_progress:
        results = results[:-1]
    frames = [pd.DataFrame(i) for i in results]
    frame = pd.concat(frames).groupby(level=0).mean().iloc[1:]
    frame["smac_train_mean"].plot()
    frame["hyperboost_train_mean"].plot()
    plt.legend()
    plt.show()
    print()