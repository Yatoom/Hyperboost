import json
import os
import time
from copy import copy

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score
from smac.scenario.scenario import Scenario

import experiments.benchmarks.config as cfgfile
from experiments.benchmarks.param_spaces import ParamSpace


def create_scenario(cs: ConfigurationSpace, deterministic: bool, run_obj: str = "quality", runcount_limit: int = 250,
                    intens_min_chall: int = 2, maxR: int = 5, output_dir="smac_output", **kwargs) -> Scenario:
    """
    Create a Scenario object

    Parameters
    ----------
    output_dir: str
        Specifies the output-directory for all emerging files, such as logging and results.
    maxR: int
        Maximum number of calls per configuration. Default: 2000.
    intens_min_chall: int
        Number of challengers to run against each other
    cs: ConfigurationSpace
        The configuration space to use within the scenario
    runcount_limit: int
        Maximum number of algorithm-calls during optimization. Default: inf
    run_obj: [‘runtime’, ‘quality’]
        Defines what metric to optimize. When optimizing runtime, cutoff_time is required as well.
    deterministic: bool
        Whether the target algorithm is

    Returns
    -------
    scenario: Scenario
        A scenario object

    """
    return Scenario({
        "run_obj": run_obj,
        "cs": cs,
        "deterministic": "true" if deterministic else "false",
        "runcount_limit": runcount_limit,
        "intens_min_chall": intens_min_chall,
        "maxR": maxR,
        # "output_dir": output_dir,
        **kwargs
    })


def create_target_algorithm_tester(param_space: ParamSpace, X_train, y_train, cv, fit_params=None, scoring=None):
    """
    Create a runner that tries out a given configuration on the training set.

    Parameters
    ----------
    param_space: ParamSpace
        The model and parameter space
    X_train: nd-array
        Training values
    y_train: 1d-array
        Training labels
    cv: int, cross-validation generator or an iterable
        Cross-validation splits to use for evaluating the algorithm
    fit_params: keyword arguments
        Parameters to pass to the fit method of the estimator
    scoring: string, callable or None
        Scoring method to use for evaluating the algorithm

    Returns
    -------
    tat: function
        The target algorithm tester
    """

    # Default setting for fit_params
    if fit_params is None:
        fit_params = {}

    # Create the target algorithm tester
    def tat(cfg, seed=None):
        # Print a dot for every time the TAT runner is executed
        print(".", end="", flush=True)

        # Initialize algorithm
        algorithm = param_space.initialize_algorithm(random_state=seed, **cfg)

        # Print a comma to indicate model initialization completed
        print(",", end="", flush=True)

        # Perform cross validation
        score = cross_val_score(algorithm, X_train, y_train, n_jobs=-1, cv=cv, fit_params=fit_params, scoring=scoring)

        # Print a semicolon to indicate cross validation finished
        print(";", end="", flush=True)

        return 1 - np.mean(score)

    return tat


def create_target_algorithm_evaluator(param_space: ParamSpace, seeds, X_train, y_train, X_test, y_test, scoring=None):
    """
    Create a runner that firstly sets the parameters for the algorithm; and secondly for each seed, retrains the
    algorithm on the training set and tries out the resulting model on the test set.

    Parameters
    ----------
    param_space: ParamSpace
        The model and parameter space
    seeds: list
        The seeds that will create reproducible runs
    X_train: nd-array
        Training values
    y_train: 1d-array
        Training labels
    X_test: nd-array
        Testing values
    y_test: 1d-array
        Testing labels
    scorer: str
        The scorer to use to validate the model

    Returns
    -------
    tae: function
        The target algorithm evaluator
    """
    scorer = get_scorer(scoring)

    def tae(cfg):
        results = []
        for seed in seeds:
            ml_algorithm = param_space.initialize_algorithm(random_state=seed, **cfg)
            ml_algorithm.fit(X_train, y_train)
            results.append(scorer(ml_algorithm, X_test, y_test))
        loss = 1 - np.mean(results)
        std = np.std(results)
        return loss, std

    return tae


def get_smac_trajectories(smac, target_algorithm_evaluator, speed=1):
    """
    Transform SMAC's incumbent update history into:
    1. A historical line of the train performance, by taking SMAC's run history
    2. A historical line of the test performance, by evaluating the incumbents on the test set

    Parameters
    ----------
    smac: SMAC
        The fitted SMAC(-like) algorithm
    target_algorithm_evaluator: function
        The target algorithm evaluator for evaluating the algorithm on the test set
    Returns
    -------
    train_trajectory: list
        A historical line of the train performance
    test_trajectory: list
        A historical line of the test performance
    """

    num_iterations = cfgfile.NUM_ITER
    train_trajectory = np.zeros(num_iterations * speed)
    test_trajectory = np.zeros(num_iterations * speed)

    # SMAC's trajectory keeps track of when the best configuration (the incumbent) is swapped out for a better one.
    # It keeps a record of the configuration, the number of runs before the swap, and the average train performance.
    for entry in smac.trajectory:
        # Get the configuration that was tried
        config = entry.incumbent._values

        # Get the number of total target algorithm runs before the swap.
        # We can use this as an index.
        num_runs = entry.ta_runs

        # Get the average performance of this configuration on the training set
        train_performance = entry.train_perf

        # We update the array with the new best value after the swap
        train_trajectory[num_runs:] = train_performance

        # Evaluate the model on the test set
        loss, std = target_algorithm_evaluator(config)
        test_trajectory[num_runs:] = loss

    return train_trajectory.tolist(), test_trajectory.tolist()


def store_json(data, name, prefix="benchmark", trial=None):
    """
    Store benchmark data in a JSON file.

    Parameters
    ----------
    data: dict
        The data to store in a JSON file
    name: str
        The name of the JSON file
    prefix: str
        The prefix to add in front of the name
    trial: int
        The seed or trial number
    """

    filename = f"{prefix}-{name}-{trial}.json"
    exists = os.path.isfile(filename)
    print(f"Stored results in {filename} ({'existing' if exists else 'new'})")

    # Initialize dictionary
    all_data = {}

    # Add existing data to dictionary if it exists
    if exists:
        with open(filename, 'r') as file:
            all_data = json.load(file)

    # Add new data to dictionary
    all_data.update(data)
    with open(filename, 'w') as file:
        json.dump(all_data, file, indent=4)


def run_smac_based_optimizer(hpo, tae, speed=1):
    hpo = copy(hpo)
    hpo.solver.intensifier.tae_runner.use_pynisher = False
    hpo.scenario.ta_run_limit = hpo.scenario.ta_run_limit * speed

    t0 = time.time()
    incumbent = hpo.optimize()
    t1 = time.time()
    train_trajectory, test_trajectory = get_smac_trajectories(hpo, tae, speed=speed)

    hpo_result = {
        "loss_train": np.array(train_trajectory).reshape(-1, speed).min(axis=1).tolist(),
        "loss_test": test_trajectory[::speed],
        "total_time": hpo.stats.wallclock_time_used,
        "run_time": hpo.stats.ta_time_used,
        "n_configs": hpo.runhistory._n_id,
    }

    info = {
        "time": t1 - t0,
        "incumbent": incumbent,
        "last_train_loss": train_trajectory[-1],
        "last_test_loss": test_trajectory[-1]
    }

    return hpo_result, info


def write_output(*args, filename="output.txt", **kwargs):
    with open(filename, "a+") as f:
        f.write(*args)


def add_record(records, task_id, name, hpo_result):
    str_task_id = str(task_id)

    if str_task_id not in records:
        records[str_task_id] = {}

    if name not in records[str_task_id]:
        records[str_task_id][name] = []

    records[str_task_id][name].append(hpo_result)

    return records
