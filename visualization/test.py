from visualization import Group, File, Collection
import pandas as pd
import os
import json

c = Collection()
c = c.add_files('../baseline/')
# c = c.add_files('.x./output/results/')
c.get_wins()
c.visualize_wins()
# c.result_table("baseline/benchmark-smac")

# print(os.getcwd())
# with open("../output/results/benchmark-RandomForest-2519249986.json", "r") as f:
    # data = json.load(f)
    # result = {
    #     seed: {
    #         optimizer: [iteration['loss_train'][-1] for iteration in results]
    #         for optimizer, results in values.items()
    #     }
    #     for seed, values in data.items()
    # }
# print(result)