import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_visualize import group
from benchmark_visualize.util import mean_of_runs, rank_against, mean_of_datasets

r = group.collect_combine_prefix("../benchmarks", "HPB-")
r = dict(zip(range(len(r)), r))
r = {seed: mean_of_runs(i) for seed, i in r.items()}
# r = {seed: rank_against(i) for seed, i in r.items()}
r = {seed: mean_of_datasets(i) for seed, i in r.items()}
r = [r[i] for i in r]
frames = [pd.DataFrame(i) for i in r]
mean = pd.concat(frames).groupby(level=0).mean().iloc[1:]
std = pd.concat(frames).groupby(level=0).std().iloc[1:]
columns = [i for i in mean.columns if "mean_train" in i]

for i in columns:
    mean[i].plot()
    plt.fill_between(np.arange(mean[i].shape[0]), mean[i] - std[i], mean[i] + std[i], alpha=0.5)

plt.legend()
plt.show()