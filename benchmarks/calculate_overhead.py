import json
import os
import numpy as np

def get_files(paths):
    for path in paths:
        with open(path, "r") as f:
            yield json.load(f)

def calculate_methods(directory):
    paths = [os.path.join(directory, i) for i in os.listdir(directory)]
    files = [i for i in get_files(paths)]

    medians = []
    for file in files:
        result = {}
        for did, ddata in file.items():
            for method, mdata in ddata.items():
                if method not in result:
                    result[method] = []
                meanie = np.mean([i['total_time'] for i in mdata])
                result[method].append(meanie)
        result = {i: [np.mean(np.array(j) / np.array(result["smac"]))] for i,j in result.items()}
        medians.append(result['hyperboost-qrd'])

    print(np.mean(medians), np.std(medians))

calculate_methods("final")