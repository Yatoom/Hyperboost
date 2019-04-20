import json
import os

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit, cross_val_score
from smac.scenario.scenario import Scenario

SEEDS = [2268061101, 2519249986, 338403738, 149713947, 523328522, 116041286, 496819236, 858316427, 272173521, 993691844]
NUM_ITER = 250
TASKS = [125920, 49, 146819, 29, 15, 3913, 3, 10101, 9971, 146818, 3917, 37, 3918, 14954, 9946, 146820, 3021, 31, 10093,
         3902, 3903, 9952, 9957, 167141, 14952, 9978, 3904, 43, 219, 14965, 7592]


def get_scenario(cs, num_iterations, deterministic):
    return Scenario({
        "run_obj": "quality",  # we optimize quality (alternatively runtime)
        "cs": cs,  # configuration space
        "deterministic": "true" if deterministic else "false",
        "runcount-limit": num_iterations,
        "intens_min_chall": 2,
        "maxR": 5,
    })


def create_smac_runner(model, X, y, cv, fit_params=None):

    if fit_params is None:
        fit_params = {}

    # Executor
    def execute_from_cfg(cfg, seed=None):
        print(".", end="", flush=True)
        mdl = from_cfg(model, cfg, seed=seed)
        ss = ShuffleSplit(n_splits=cv, random_state=0, test_size=0.10, train_size=None)
        mdl.fit(X, y)
        acc = cross_val_score(mdl, X, y, cv=ss, n_jobs=-1, fit_params=fit_params)
        return 1 - np.mean(acc)

    return execute_from_cfg


def from_cfg(model, cfg, seed=None):
    config = {k: cfg[k] for k in cfg}
    config = {k: True if i == "True" else False if i == "False" else i for k, i in config.items()}
    config = {k: None if i == "None" else i for k, i in config.items()}

    # print(config)
    try:
        mdl = model.model(random_state=seed, **config)
    except:
        mdl = model.model(**config)
    return mdl


def get_smac_trajectories(smac, model, num_iterations, X_train, y_train, X_test, y_test, seeds):
    test_trajectory = np.zeros(num_iterations)
    train_trajectory = np.zeros(num_iterations)
    for entry in smac.trajectory:
        index = entry.ta_runs
        train_perf = entry.train_perf
        config = entry.incumbent._values
        loss, std = validate_model(model, config, X_train, y_train, X_test, y_test, seeds)
        train_trajectory[index:] = train_perf
        test_trajectory[index:] = loss
    return train_trajectory.tolist(), test_trajectory.tolist()


def validate_model(model, best_config, X_train, y_train, X_test, y_test, seeds):
    accuracies = []
    for seed in seeds:
        mdl = from_cfg(model, best_config, seed)
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    loss = 1 - np.mean(accuracies)
    std = np.std(accuracies)
    return loss, std


def store_json(data, name, trial=None):
    filename = f"results-{name}-{trial}.json"

    exists = os.path.isfile(filename)
    all_data = {}
    if exists:
        with open(filename, 'r') as file:
            all_data = json.load(file)

    all_data.update(data)
    with open(filename, 'w') as file:
        # print(all_data)
        json.dump(all_data, file, indent=4)
