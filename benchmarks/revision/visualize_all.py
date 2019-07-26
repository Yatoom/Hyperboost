import json
import os
import re
import pandas as pd
import numpy as np
import benchmark_visualize.util as util
import matplotlib.pyplot as plt

plt.style.use("seaborn")
DIR = ".."
PREFIX = "AVOID-"


def rename(i):
    n = i.replace("hyperboost", "Hyperboost")
    n = n.replace("ei2", "EI")
    n = n.replace("_2x", " 2x")
    n = n.replace("_mean_train", "")
    n = n.replace("_mean_test", "")
    n = n.replace("roar", "ROAR")
    n = n.replace("smac", "SMAC")
    n = n.replace("random", "Random")
    n = n.replace("-std-y", "")
    n = n.replace("pca", "PCA")
    return n


grouped = {}
for file in os.listdir(DIR):
    if file.startswith(PREFIX):
        postfix = re.search(r"[0-9]+", file)[0]
        base = file.replace(f"-{postfix}", "")
        base = base.replace(".json", "")
        base = base.replace("ALL-", "")

        if base not in grouped.keys():
            grouped[base] = []

        with open(os.path.join(DIR, file), "r") as f:
            data = json.load(f)

        grouped[base].append(data)

print(grouped.keys())

runs = ["hyperboost-std-y", "smac", "roar", "hyperboost-ei2", "random_2x", "roar_2x", "hyperboost-pca-std-y"]
keep = ["random_2x", "roar_2x", "smac", "hyperboost-std-y"]
# keep = ["smac", "hyperboost-std-y", "hyperboost-ei2"]
# keep = ["smac", "hyperboost-std-y", "hyperboost-pca-std-y"]

# Taking the mean of each run
for group in grouped.keys():  # SVM, DecisionTree, RandomForest
    print(group)
    for seed in range(len(grouped[group])):  # 0, 1, 2

        # Taking the mean per iteration: mean per run
        mean_runs = util.mean_of_runs(grouped[group][seed], keep_runs=keep)

        # Taking the mean over all dataset: mean per method
        grouped[group][seed] = util.mean_of_datasets(mean_runs)

    # Taking the mean over all seeds
    frames = [pd.DataFrame(i) for i in grouped[group]]
    mean = pd.concat(frames).groupby(level=0).mean().iloc[1:]
    std = pd.concat(frames).groupby(level=0).std().iloc[1:]

    # Visualize train
    columns = [i for i in mean.columns if "mean_train" in i]

    max_100 = 0
    min_100 = 1
    renamed_columns = []
    for i in columns:
        n = rename(i)

        # if n in ["Random 2x", "ROAR 2x", "ROAR", "Hyperboost-EI"]:
        #     continue
        # if n in ["ROAR", "Hyperboost-PCA", "Hyperboost-EI"]:
        #     continue

        # max_100 = max(max_100, mean[i].iloc[50] + 2 * std[i].iloc[50])
        # min_100 = min(min_100, mean[i].iloc[247] - 2 * std[i].iloc[247])
        renamed_columns.append(n)
        mean[i].plot()
        plt.fill_between(np.arange(mean[i].shape[0]), mean[i] - std[i], mean[i] + std[i], alpha=0.5)
    # plt.ylim(bottom=min_100, top=max_100)
    # plt.ylim(1, 4)
    plt.xlabel("Iteration")
    plt.ylabel("Ranking (lower is better)")
    plt.legend(renamed_columns)
    plt.show()

print()
