import json
import os
import re

from benchmarks.config import dict_raise_on_duplicates


def combine_two_dicts(original, additional, keys):
    result = original
    for key in keys:
        result = {i: dict(j, **{key: additional[i][key] if key in additional[i] else None}) for i, j in result.items() if i in additional.keys()}
    return result


def combine_two_files(original, additional, keys):
    with open(original, "r") as f:
        original = json.load(f, object_pairs_hook=dict_raise_on_duplicates)
    with open(additional, "r") as f:
        additional = json.load(f, object_pairs_hook=dict_raise_on_duplicates)
    result = combine_two_dicts(original, additional, keys=keys)
    # result = {i: dict(j, **{key: additional[i][key]}) for i, j in original.items()}
    return result


def matches(name, filename):
    match = re.match(fr"results-{name}-[0-9]*\.json", filename)
    return False if match is None else True

def get_seed(filename):
    return re.findall(r"[0-9]+", filename)[0]

def combine_all(o, a, keys):
    originals = [os.path.join(o["dir"], filename) for filename in os.listdir(o['dir']) if matches(o['name'], filename)]
    additionals = [os.path.join(a["dir"], filename) for filename in os.listdir(a['dir']) if matches(a['name'], filename)]

    # Get seeds
    o_seeds = [get_seed(i) for i in originals]
    a_seeds = [get_seed(i) for i in originals]

    # Get intersection of both sets
    intersection = set(o_seeds + a_seeds)
    originals = [filepath for seed, filepath in zip(o_seeds, originals) if seed in intersection]
    additionals = [filepath for seed, filepath in zip(o_seeds, additionals) if seed in intersection]

    print("Original:", originals)
    print("Additions:", additionals)

    # Combine files to retrieve the result
    result = {seed: combine_two_files(i, j, keys) for i, j, seed in zip(originals, additionals, intersection)}
    return result


if __name__ == "__main__":
    r = combine_all({"dir": "LightQR", "name": "DecisionTree-stoch"},
                    {"dir": "LightQR", "name": "DecisionTree-stoch-QRD"},
                    ["hyperboost-qrd"])

    # f1 = "LightQR/results-DecisionTree-stoch-2268061101.json"
    # f2 = "LightQR/results-DecisionTree-stoch-QRD-2268061101.json"
    # r = combine_two_files(f1, f2, key="hyperboost-qrd")
    print()
