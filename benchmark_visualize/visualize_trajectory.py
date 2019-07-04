import json
import os
import re
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from benchmarks import config

# Settings
from benchmarks.combine_results import combine_all

plt.style.use("seaborn")
dir = "HyperBoostEI"
in_progress = True
ranking = True
NAME = "RandomForest"


def matches(filename):
    match = re.match(fr"results-{NAME}-[0-9]*\.json", filename)
    return False if match is None else True


files = [filename for filename in os.listdir(dir) if matches(filename)]
print("matching files:", files)




def load(f):
    # return json.loads(f)
    with open(f, "r") as file:
        return json.load(file)

# if __name__ == "__main__":
#     for instance in ["train", "test"]:
#         for model, shorthand in zip(["RandomForest"], ["rf"]):
#             r = {filename: load(os.path.join(dir, filename)) for filename in os.listdir(dir) if matches(filename)}
#             # r = {0: r}
#             # r = combine_all({"dir": ".", "name": model},
#             #                 {"dir": "final-random", "name": model},
#             #                 ["random_2x", "hyperboost-qrd", "hyperboost-std"])
#             # r = {i: r[i] for i in range(len(r))}
#             r = {seed: mean_of_runs(i) for seed, i in r.items()}
#             if ranking:
#                 r = {seed: rank_against(i) for seed, i in r.items()}
#             else:
#                 r = {seed: mean_of_datasets(i) for seed, i in r.items()}
#             results = r
#
#             # results = {}
#             # for file in files:
#             #     seed = re.findall(r"[0-9]+", file)[0]
#             #     with open(os.path.join(dir, file), "r") as f:
#             #         data = json.load(f)
#             #         keep_runs = ["smac", "hyperboost-drop"]
#             #         r = mean_of_runs(data, keep_runs=None)
#             #         if ranking:
#             #             r = rank_against(r, dual=False)
#             #         else:
#             #             r = mean_of_datasets(r)
#             #         results[seed] = r
#
#             # results = [results[str(i)] for i in config.SEEDS if str(i) n results.keys()]
#             results = list(results.values())
#             # if in_progress:
#             #     results = results[:-1]
#             frames = [pd.DataFrame(i) for i in results]
#             mean = pd.concat(frames).groupby(level=0).mean().iloc[1:]
#             std = pd.concat(frames).groupby(level=0).std().iloc[1:]
#
#             columns = [i for i in mean.columns if f"mean_{instance}" in i]
#             # columns = ["smac_mean_train", "hyperboost-qrd_mean_train"]
#
#             plt.clf()
#             for i in columns:
#                 name = i.replace("hyperboost-qrd", "Hyperboost")
#                 name = name.replace("smac", "SMAC")
#                 name = name.replace("random_2x", "Random x2")
#                 name = name.replace("_test", "")
#                 name = name.replace("_train", "")
#                 name = name.replace("_mean", "")
#                 mean[i].plot(label=name)
#                 plt.fill_between(np.arange(mean[i].shape[0]), mean[i] - std[i], mean[i] + std[i], alpha=0.5)
#
#             plt.legend()
#             # plt.title(name)
#             plt.xlabel("Iterations")
#             if ranking:
#                 plt.ylabel("Ranking (lower is better)")
#                 plt.ylim(1, 3)
#             else:
#                 plt.ylabel("Mean loss")
#             # plt.show()
#             plt.savefig(f"{shorthand}-{instance}.png")
#             print(shorthand, instance)


if __name__ == "__main__":
    r = combine_all({"dir": "final", "name": "RandomForest"},
                    {"dir": "HyperBoostEINew", "name": "RandomForest"},
                    ["hyperboost-ei"])
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
    #
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