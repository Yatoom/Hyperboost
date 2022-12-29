import os
import time
from rich.progress import track, MofNCompleteColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, \
    TimeElapsedColumn
import numpy as np
import openml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from benchmark import config, util
from benchmark.util import write_output, run_smac_based_optimizer
from hyperboost.epm.catboost_epm import CatboostEPM
# from hyperboost.hyperboost import Hyperboost
from smac.facade.roar_facade import ROAR
from smac.facade.smac_hpo_facade import SMAC4HPO
from rich.progress import Progress

# Create output directory
if not os.path.exists(config.BASE_DIRECTORY):
    os.mkdir(config.BASE_DIRECTORY)

# Benchmark
progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeRemainingColumn(),
    TimeElapsedColumn(),
)
with progress as progress:

    task1 = progress.add_task("[red]Iteration", total=len(config.SEEDS))
    task2 = progress.add_task("[green]Target algo", total=len(config.ML_ALGORITHMS))
    task3 = progress.add_task("[cyan]Task", total=len(config.TASKS))
    task4 = progress.add_task("[yellow]Outer loop", total=config.N_SPLITS)
    task5 = progress.add_task("[purple]Optimizer", total=3)
    task6 = progress.add_task("[white]Train/test", total=1)
    task7 = progress.add_task("[red]Evaluations", total=config.NUM_ITER)

    for hpo_state in config.SEEDS:

        # Setup random number generator
        rng = np.random.RandomState(hpo_state)

        for ml_algorithm in config.ML_ALGORITHMS:

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
                scenario_1x, scenario_2x = util.create_scenario(ml_algorithm.configuration_space, ml_algorithm.is_deterministic,
                                                subpath=[str(hpo_state), ml_algorithm.name, str_task_id])

                # Log to output
                write_output(f"Task {task_id}\n")

                # for task_id in track(config.TASKS, description='[red]Task'):

                for train_index, test_index in config.OUTER_LOOP.split(X, y):

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
                    tat = util.create_target_algorithm_tester(ml_algorithm, X_train, y_train, cv=config.INNER_LOOP,
                                                              scoring=config.METRIC, progress=progress, eval_tracker=task7)
                    tae = util.create_target_algorithm_evaluator(ml_algorithm, config.SEEDS, X_train, y_train, X_test,
                                                                 y_test, scoring=config.METRIC)

                    ########################################################################################################
                    # Hyperboost
                    ########################################################################################################
                    name = "hyperboost"
                    progress.console.print(f"Iteration seed = {hpo_state} | Target algo = {ml_algorithm.name} | Task id = {task_id} | Outer loop = {task4.denominator} | Optimizer = {name}")
                    hpo = SMAC4HPO(scenario=scenario_1x, rng=rng, tae_runner=tat, model=CatboostEPM)
                    hpo_result, info = run_smac_based_optimizer(hpo, tae, progress=progress, stage_tracker=task6)
                    progress.console.print(
                        f"> Time = {info['time']} | Run Time = {hpo_result['run_time']} | Clock = {hpo_result['total_time']} |"
                        f"Train loss = {info['last_train_loss']} | Test loss = {info['last_test_loss']}"
                    )

                    records = util.add_record(records, task_id, name, hpo_result)
                    progress.update(task5, advance=1)
                    progress.reset(task7)

                    ########################################################################################################
                    # SMAC
                    ########################################################################################################
                    name = "smac"
                    progress.console.print(f"Iteration seed = {hpo_state} | Target algo = {ml_algorithm.name} | Task id = {task_id} | Outer loop = {task4.denominator} | Optimizer = {name}")
                    hpo = SMAC4HPO(scenario=scenario_1x, rng=rng, tae_runner=tat)
                    hpo_result, info = run_smac_based_optimizer(hpo, tae, progress=progress, stage_tracker=task6)
                    progress.console.print(
                        f"> Time = {info['time']} | Run Time = {hpo_result['run_time']} | Clock = {hpo_result['total_time']} |"
                        f"Train loss = {info['last_train_loss']} | Test loss = {info['last_test_loss']}"
                    )

                    records = util.add_record(records, task_id, name, hpo_result)
                    progress.update(task5, advance=1)
                    progress.reset(task7)

                    ########################################################################################################
                    # ROAR x2
                    ########################################################################################################
                    name = "roar_x2"
                    progress.console.print(f"Iteration seed = {hpo_state} | Target algo = {ml_algorithm.name} | Task id = {task_id} | Outer loop = {task4.denominator} | Optimizer = {name}")
                    hpo = ROAR(scenario=scenario_2x, rng=rng, tae_runner=tat)
                    hpo_result, info = run_smac_based_optimizer(hpo, tae, progress=progress, stage_tracker=task6)
                    progress.console.print(
                        f"> Time = {info['time']} | Run Time = {hpo_result['run_time']} | Clock = {hpo_result['total_time']} |"
                        f"Train loss = {info['last_train_loss']} | Test loss = {info['last_test_loss']}"
                    )

                    records = util.add_record(records, task_id, name, hpo_result)
                    progress.update(task5, advance=1)
                    progress.reset(task7)

                    ########################################################################################################
                    # Random
                    ########################################################################################################
                    # name = "random_x2"
                    # progress.console.print(f"Iteration seed = {hpo_state} | Target algo = {ml_algorithm.name} | Task id = {task_id} | Outer loop = {task4.denominator} | Optimizer = {name}")
                    # speed = 2
                    # best_loss = 1
                    # last_test_loss = None
                    # train_trajectory = []
                    # test_trajectory = []
                    # running_time = 0
                    # for i in range(config.NUM_ITER):
                    #     start = time.time()
                    #     configs = [ml_algorithm.configuration_space.sample_configuration() for _ in range(speed)]
                    #     losses = [tat(cfg) for cfg in configs]
                    #     best = np.argmin(losses)
                    #     end = time.time()
                    #     running_time += end - start
                    #     if losses[best] < best_loss:
                    #         best_loss = losses[best]
                    #         test_loss, test_std = tae(configs[best])
                    #         last_test_loss = test_loss
                    #     test_trajectory.append(last_test_loss)
                    #     train_trajectory.append(best_loss)
                    #
                    # hpo_result = {
                    #     "loss_train": train_trajectory,
                    #     "loss_test": test_trajectory,
                    #     "total_time": running_time / speed,
                    #     "run_time": running_time,
                    #     "n_configs": config.NUM_ITER * speed,
                    # }
                    #
                    # records = util.add_record(records, task_id, name, hpo_result)
                    # progress.update(task5, advance=1)
                    # progress.reset(task7)

                    ########################################################################################################

                    write_output("\n")

                    progress.update(task4, advance=1)
                    progress.reset(task5)

                # Store results
                util.store_json(records, name=ml_algorithm.name, trial=hpo_state)

                progress.update(task3, advance=1)
                progress.reset(task4)
            progress.update(task2, advance=1)
            progress.reset(task3)
        progress.update(task1, advance=1)
        progress.reset(task2)
