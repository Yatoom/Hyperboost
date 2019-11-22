import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark import group, config
from benchmark.visualize_util import mean_of_runs, mean_of_datasets, rank_against

plt.style.use("seaborn")

# Collect the files from the results directory
r = group.collect_combine_prefix(config.RESULTS_DIRECTORY, config.RESULTS_PREFIX + "-")
r = dict(zip(range(len(r)), r))
r = {seed: mean_of_runs(i) for seed, i in r.items()}

# Comment the first line, and uncomment the second line to show ranking
r = {seed: mean_of_datasets(i) for seed, i in r.items()}
# r = {seed: rank_against(i) for seed, i in r.items()}

# Aggregate results
r = [r[i] for i in r]
frames = [pd.DataFrame(i) for i in r]
mean = pd.concat(frames).groupby(level=0).mean().iloc[1:].reset_index(drop=True)
std = pd.concat(frames).groupby(level=0).std().iloc[1:].reset_index(drop=True)

# Filter on columns with the word "mean_train" in it. Same works for "mean_test".
columns = [i for i in mean.columns if "mean_test" in i]


# We can also filter for one specific target model
# columns = [i for i in columns if "DecisionTree" in i]

# Here we can rename our columns.
def rename(column):
    column = column.replace("_mean_train", "")
    column = column.replace("hyperboost", "HyperBoost")
    column = column.replace("smac", "SMAC")
    column = column.replace(config.RESULTS_PREFIX + "-", "")
    return column


# Plot the results
for i in columns:
    mean[i].plot(label=rename(i))
    plt.fill_between(np.arange(mean[i].shape[0]), mean[i] - std[i], mean[i] + std[i], alpha=0.5)

plt.legend()
plt.show()
