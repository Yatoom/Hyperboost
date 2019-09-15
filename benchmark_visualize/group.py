import json
import os
import re


def collect_filenames(folder, prefix):
    """
    Get list of filenames in folder that start with prefix.
    :param folder: folder to look in
    :param prefix: prefix string
    :return: maching filenames
    """
    collection = os.listdir(folder)
    matches = [i for i in collection if i.startswith(prefix)]
    return matches


def group_by_prefix(filenames):
    """
    Group filenames by prefix.
    :param filenames: prefix-[0-9].json
    :return: prefix > prefix-[0-9].json
    """
    groups = [re.sub(r"-[0-9]*.json", "", i) for i in filenames]
    layout = {}
    for g, m in zip(groups, filenames):
        if g not in layout:
            layout[g] = []
        layout[g].append(m)
    return layout


def load_layout(layout, folder):
    """
    Load data from filenames specified in layout.
    :param layout: prefix > filenames
    :param folder: location where filenames reside
    :return: prefix > iteration > task_id > method > run_results
    """
    data = {}
    for key, files in layout.items():
        results = []
        for file in files:
            with open(os.path.join(folder, file)) as f:
                results.append(json.load(f))
        data[key] = results
    return data


def prefix_data(multi_data):
    """
    Prefix methods and remove missing results.
    :param multi_data: prefix > iteration > task_id > method > run_results
    :return: iteration > task_id > prefixed_method > run_results
    """
    cleaned_data = []
    for filename, iterations in multi_data.items():
        for iter_index, iteration in enumerate(iterations):
            for task, runs, in iteration.items():
                for method, run_results in runs.items():
                    if len(cleaned_data) <= iter_index:
                        cleaned_data.append({})
                    if task not in cleaned_data[iter_index]:
                        cleaned_data[iter_index][task] = {}
                    if len(run_results) > 0:
                        cleaned_data[iter_index][task][f"{filename}-{method}"] = run_results

    return cleaned_data


def collect_combine_prefix(folder, prefix):
    """
    Collect files in folder with prefix, combine them together and prefix the methods.
    :param folder: folder to look in
    :param prefix: prefix string
    :return: iteration > task_id > prefixed_method > run_results
    """
    matches = collect_filenames(folder, prefix)
    layout = group_by_prefix(matches)
    multi_data = load_layout(layout, FOLDER)
    result = prefix_data(multi_data)
    return result


if __name__ == "__main__":
    PREFIX = "HPB-"
    FOLDER = "../benchmarks"

    matches = collect_filenames(FOLDER, PREFIX)
    layout = group_by_prefix(matches)
    multi_data = load_layout(layout, FOLDER)
    result = prefix_data(multi_data)
    print()
