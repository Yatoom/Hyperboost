import numpy as np

import matplotlib.pyplot as plt

from benchmark_visualize.util import load, mean_of_runs, mean_of_datasets, rename, rank_against

plt.style.use("seaborn")
filename = "../benchmarks/HPB-005-GBM-2268061101.json"
data = load(filename)
excluded = "hyperboost"
data = {task_id: {method: values for method, values in results.items() if method not in excluded} for task_id, results in data.items()}

r = mean_of_runs(data)
r = mean_of_datasets(r)
# r = rank_against(r)

train = {rename(k): v for k, v in r.items() if "mean_train" in k}
train_std = {rename(k): v for k, v in r.items() if "std_train" in k}
test = {rename(k): v for k, v in r.items() if "mean_test" in k}
test_std = {rename(k): v for k, v in r.items() if "std_test" in k}

for k, v in train.items():
    # if not isinstance(v, np.ndarray):
    #     continue
    mean = v[1:]
    std = train_std[k][1:]
    x = np.arange(len(v) - 1)
    plt.plot(mean, label=k)
    # plt.fill_between(x, mean - std, mean + std)
plt.legend()
plt.show()
print()
