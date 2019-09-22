import time

import numpy as np
import openml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, ShuffleSplit

from experiments.benchmarks import config, util
from experiments.benchmarks.param_spaces import RandomForestSpace
from experiments.benchmarks.util import write_output, run_smac_based_optimizer
from hyperboost.hyperboost import Hyperboost
from smac.facade.roar_facade import ROAR
from smac.facade.smac_facade import SMAC

# Options
ml_algorithms = [RandomForestSpace()]
outer_loop = KFold(n_splits=3, shuffle=True, random_state=42)
inner_loop = ShuffleSplit(n_splits=3, random_state=0, test_size=0.10, train_size=None)

for hpo_state in config.SEEDS:

    # Setup random number generator
    rng = np.random.RandomState(hpo_state)

    for ml_algorithm in ml_algorithms:
        records = {}

        for task_id in config.TASKS:

            # Convert task_id to string to avoid key-conflicts in JSON
            str_task_id = str(task_id)
            records[str_task_id] = {}

            # Get task data from OpenML
            task = openml.tasks.get_task(task_id)
            X, y = task.get_X_and_y()
            dataset = task.get_dataset()
            categorical = dataset.get_features_by_type("nominal", exclude=[task.target_name])
            numeric = dataset.get_features_by_type("numeric", exclude=[task.target_name])

            # Create a scenario object for SMAC
            scenario = util.create_scenario(ml_algorithm.configuration_space, ml_algorithm.is_deterministic,
                                            runcount_limit=config.NUM_ITER)

            # Log to output
            write_output(f"Task {task_id}")

            for train_index, test_index in outer_loop.split(X):
                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]

                # Fill missing values using different strategies for categorical and numeric parameters
                ct = ColumnTransformer([
                    ["most_frequent", SimpleImputer(strategy="most_frequent"), categorical],
                    ["median", SimpleImputer(strategy="median"), numeric]
                ])

                X_train = ct.fit_transform(X_train)
                X_test = ct.transform(X_test)

                # Setup evaluator and tester
                tat = util.create_target_algorithm_tester(ml_algorithm, X_train, y_train, cv=inner_loop,
                                                          scoring="balanced_accuracy")
                tae = util.create_target_algorithm_evaluator(ml_algorithm, config.SEEDS, X_train, y_train, X_test,
                                                             y_test, scoring="balanced_accuracy")

                ########################################################################################################
                # Hyperboost
                ########################################################################################################
                hpo = Hyperboost(scenario=scenario, rng=rng, tae_runner=tat, pca_components=2)
                hpo_result, info = run_smac_based_optimizer(hpo, tae)
                name = "hyperboost"

                write_output(f"[{name}] time={info['time']} train_loss={info['last_train_loss']} "
                             f"test_loss={info['last_test_loss']}\n")

                records = util.add_record(records, task_id, name, hpo_result)

                ########################################################################################################
                # SMAC
                ########################################################################################################
                hpo = SMAC(scenario=scenario, rng=rng, tae_runner=tat, use_pynisher=False)
                hpo_result, info = run_smac_based_optimizer(hpo, tae)
                name = "smac"

                write_output(f"[{name}] time={info['time']} train_loss={info['last_train_loss']} "
                             f"test_loss={info['last_test_loss']}\n")

                records = util.add_record(records, task_id, name, hpo_result)

                ########################################################################################################
                # ROAR x2
                ########################################################################################################
                hpo = ROAR(scenario=scenario, rng=rng, tae_runner=tat, use_pynisher=False)
                hpo_result, info = run_smac_based_optimizer(hpo, tae, speed=2)
                name = "roar_x2"

                write_output(f"[{name}] time={info['time']} train_loss={info['last_train_loss']} "
                             f"test_loss={info['last_test_loss']}\n")

                records = util.add_record(records, task_id, name, hpo_result)

                ########################################################################################################
                # Random
                ########################################################################################################
                SPEED = 2
                best_loss = 1
                last_test_loss = None
                train_trajectory = []
                test_trajectory = []
                running_time = 0
                for i in range(config.NUM_ITER):
                    start = time.time()
                    configs = [ml_algorithm.cs.sample_configuration() for _ in range(SPEED)]
                    losses = [tat(cfg) for cfg in configs]
                    best = np.argmin(losses)
                    end = time.time()
                    running_time += end - start
                    if losses[best] < best_loss:
                        best_loss = losses[best]
                        test_loss, test_std = config.validate_model(ml_algorithm, configs[best], X_train, y_train,
                                                                    X_test, y_test, config.SEEDS)
                        last_test_loss = test_loss
                    test_trajectory.append(last_test_loss)
                    train_trajectory.append(best_loss)

                hpo_result = {
                    "loss_train": train_trajectory,
                    "loss_test": test_trajectory,
                    "total_time": running_time / SPEED,
                    "run_time": running_time,
                    "n_configs": config.NUM_ITER * SPEED,
                }

                records = util.add_record(records, task_id, name, hpo_result)

                ########################################################################################################

            # Store results
            util.store_json(records, ml_algorithm.name, hpo_state)
