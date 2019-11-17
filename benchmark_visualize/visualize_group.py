import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_visualize import group
from benchmark_visualize.util import mean_of_runs, rank_against, mean_of_datasets

plt.style.use("seaborn")

def exclude(inp, exclude):
    result = []
    # iteration > task_id > prefixed_method > run_results
    for i in range(len(inp)):
        for j in inp[i]:
            for k in inp[i][j]:
                if k == exclude:
                    continue
                if len(result) <= i:
                    result.append({})
                if j not in result[i]:
                    result[i][j] = {}
                result[i][j][k] = inp[i][j][k]
    return result


for part in ["train", "test"]:
    for method in ["rank", "loss"]:
        r = group.collect_combine_prefix("../benchmarks", "SHPB-")
        r = exclude(r, "SHPB-GBM-hyperboost-pca")
        r = dict(zip(range(len(r)), r))
        r = {seed: mean_of_runs(i) for seed, i in r.items()}

        if method == "rank":
            r = {seed: rank_against(i) for seed, i in r.items()}
        elif method == "loss":
            r = {seed: mean_of_datasets(i) for seed, i in r.items()}
        r = [r[i] for i in r]
        frames = [pd.DataFrame(i) for i in r]
        mean = pd.concat(frames).groupby(level=0).mean().iloc[1:]
        std = pd.concat(frames).groupby(level=0).std().iloc[1:]
        columns = [i for i in mean.columns if f"mean_{part}" in i]

        selection = {
            f"SHPB-005-GBM-hyperboost-pca_mean_{part}": "Hyperboost",
            f"SHPB-GBM-smac_mean_{part}": "SMAC",
            f"SHPB-R-GBM-roar_2x_mean_{part}": "ROAR 2x",
            f"SHPB-R-GBM-random_2x_mean_{part}": "Random 2x"
        }
        order = ["Hyperboost", "Random 2x", "ROAR 2x", "SMAC"]
        columns = {order.index(selection[i]): i for i in columns}
        columns = [columns[i] for i in range(4)]

        for i in columns:
            # if i not in selection.keys():
            #     continue
            mean[i].plot(label=selection[i], figsize=(8, 5.5))
            plt.fill_between(np.arange(mean[i].shape[0]), mean[i] - std[i], mean[i] + std[i], alpha=0.5)

        plt.xlabel("Iterations")
        if method == "loss":
            plt.ylabel("Mean loss")
        if method == "rank":
            plt.ylabel("Ranking (lower is better)")
        plt.legend()
        # plt.show()
        plt.tight_layout()
        plt.savefig(f"gbm-{method}-{part}.png")
        plt.clf()
