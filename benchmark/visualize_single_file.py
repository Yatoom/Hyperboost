import numpy as np

import matplotlib.pyplot as plt

from benchmark.visualize_util import load, mean_of_runs, mean_of_datasets, rename

plt.style.use("seaborn")
filename = "../benchmarks/NEW-RandomForest-2268061101.json"
data = load(filename)

r = mean_of_runs(data)
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
