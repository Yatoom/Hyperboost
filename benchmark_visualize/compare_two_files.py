import numpy as np

import matplotlib.pyplot as plt

from benchmark_visualize.util import load, mean_of_runs, mean_of_datasets, rename, combine_two_files, combine_two_dicts, \
    rank_against

plt.style.use("seaborn")

data1 = load("../benchmarks/NEW-RandomForest-2268061101.json")
data2 = load("../benchmarks/NEW2-RandomForest-2268061101.json")
data3 = load("../benchmarks/NEW3-RandomForest-2268061101.json")
data12 = combine_two_dicts(data1, data2, keys=["hyperboost-ei2"])
data123 = combine_two_dicts(data12, data3, keys=["hyperboost-qrd-correct"])

# data = combine_two_files(
#     original="../benchmarks/NEW-RandomForest-2268061101.json",
#     additional="../benchmarks/NEW3-RandomForest-2268061101.json",
#     keys=["hyperboost-qrd-correct"])

r = mean_of_runs(data123)
r = mean_of_datasets(r)

train = {rename(k): v for k, v in r.items() if "mean_train" in k}
train_std = {rename(k): v for k, v in r.items() if "std_train" in k}
test = {rename(k): v for k, v in r.items() if "mean_test" in k}
test_std = {rename(k): v for k, v in r.items() if "std_test" in k}

for k, v in train.items():
    mean = v[1:]
    std = train_std[k][1:]
    x = np.arange(len(v) - 1)
    plt.plot(mean, label=k)
    # plt.fill_between(x, mean - std, mean + std)
plt.legend()
plt.show()
print()
