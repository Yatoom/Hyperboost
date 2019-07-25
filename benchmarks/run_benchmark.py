import time

import numpy as np
import openml
from sklearn.model_selection import KFold

from smac.facade.roar_facade import ROAR
from smac.facade.smac_facade import SMAC

from benchmarks import config
from benchmarks.config import create_smac_runner
from benchmarks.param_spaces import RandomForestSpace, DecisionTreeSpace, SVMSpace, LDASpace, AdaboostSpace
from benchmarks.preprocessing import ConditionalImputer
from hyperboost.hyperboost import Hyperboost


def write(*args, **kwargs):
    with open("output.txt", "a+") as f:
        f.write(*args)


for state in config.SEEDS:
    rng = np.random.RandomState(state)
    for model in [RandomForestSpace, DecisionTreeSpace, SVMSpace]:
        records = {}
        for task_id in config.TASKS:

            str_task_id = str(task_id)

            # records[task_id] = {"smac": [], "hyperboost-drop": [], "hyperboost-combo": [], "hyperboost-var": []}
            records[str_task_id] = {"hyperboost-std-y-eps-1": []}

            task = openml.tasks.get_task(task_id)
            X, y = task.get_X_and_y()
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            dataset = task.get_dataset()
            categorical = dataset.get_features_by_type("nominal", exclude=[task.target_name])
            scenario = config.get_scenario(model.cs, config.NUM_ITER, model.is_deterministic)

            write(f"Task {task_id}")

            for train_index, test_index in kf.split(X):
                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]

                ci = ConditionalImputer(categorical_features=categorical)
                X_train = ci.fit_transform(X_train)
                X_test = ci.transform(X_test)

                try_params = create_smac_runner(model, X_train, y_train, 3)

                ########################################################################################################
                # Random
                ########################################################################################################
                # SPEED = 2
                # best_loss = 1
                # last_test_loss = None
                # train_trajectory = []
                # test_trajectory = []
                # running_time = 0
                # for i in range(config.NUM_ITER):
                #     start = time.time()
                #     configs = [model.cs.sample_configuration() for _ in range(SPEED)]
                #     losses = [try_params(cfg) for cfg in configs]
                #     best = np.argmin(losses)
                #     end = time.time()
                #     running_time += end - start
                #     if losses[best] < best_loss:
                #         best_loss = losses[best]
                #         test_loss, test_std = config.validate_model(model, configs[best], X_train, y_train, X_test, y_test, config.SEEDS)
                #         last_test_loss = test_loss
                #     test_trajectory.append(last_test_loss)
                #     train_trajectory.append(best_loss)
                #
                # smac_res = {
                #     "loss_train": train_trajectory,
                #     "loss_test": test_trajectory,
                #     "total_time": running_time / SPEED,
                #     "run_time": running_time,
                #     "n_configs": config.NUM_ITER * SPEED,
                # }
                # records[task_id][f"random_{SPEED}x"].append(smac_res)
                # print()

                ########################################################################################################
                # ROAR
                ########################################################################################################
                # roar = ROAR(scenario=scenario, rng=rng, tae_runner=try_params, use_pynisher=False)
                # roar_start = time.time()
                # incumbent_roar = roar.optimize()
                # roar_end = time.time()
                # print(f"ROAR time: {roar_end - roar_start}")
                # roar_train, roar_test = config.get_smac_trajectories(roar, model, config.NUM_ITER, X_train,
                #                                                      y_train, X_test, y_test,
                #                                                      seeds=config.SEEDS)
                # write(f"\r[ROAR] train loss = {roar_train[-1]} | test loss = {roar_test[-1]} | config = {incumbent_roar._values}")
                # roar_res = {
                #     "loss_train": roar_train,
                #     "loss_test": roar_test,
                #     "total_time": roar.stats.wallclock_time_used,
                #     "run_time": roar.stats.ta_time_used,
                #     "n_configs": roar.runhistory._n_id,
                # }
                # records[str_task_id]["roar"].append(roar_res)



                ########################################################################################################
                # SMAC
                ########################################################################################################
                smac = SMAC(scenario=scenario, rng=rng, tae_runner=try_params, use_pynisher=False)
                smac_start = time.time()
                incumbent_smac = smac.optimize()
                smac_end = time.time()
                print(f"SMAC time: {smac_end - smac_start}")
                smac_train, smac_test = config.get_smac_trajectories(smac, model, config.NUM_ITER, X_train,
                                                                     y_train, X_test, y_test,
                                                                     seeds=config.SEEDS)
                write(f"\r[SMAC] train loss = {smac_train[-1]} | test loss = {smac_test[-1]} | config = {incumbent_smac._values}")
                smac_res = {
                    "loss_train": smac_train,
                    "loss_test": smac_test,
                    "total_time": smac.stats.wallclock_time_used,
                    "run_time": smac.stats.ta_time_used,
                    "n_configs": smac.runhistory._n_id,
                }
                records[str_task_id]["smac"].append(smac_res)

                # ########################################################################################################

                ########################################################################################################
                # Hyperboost With Std
                ########################################################################################################
                # hyperboost = Hyperboost(scenario=scenario, rng=rng, method="skopt", tae_runner=try_params)
                # hyper_start = time.time()
                # incumbent_hyperboost = hyperboost.optimize()
                # hyper_end = time.time()
                # print(f"HyperboostWithStd time: {hyper_end - hyper_start}")
                # hb_train, hb_test = config.get_smac_trajectories(hyperboost, model, config.NUM_ITER, X_train,
                #                                                  y_train, X_test, y_test,
                #                                                  seeds=config.SEEDS)
                # write(f"\r[HYBO] train loss = {hb_train[-1]} | test loss = {hb_test[-1]}")
                #
                # hb_res = {
                #     "loss_train": hb_train,
                #     "loss_test": hb_test,
                #     "total_time": hyperboost.stats.wallclock_time_used,
                #     "run_time": hyperboost.stats.ta_time_used,
                #     "n_configs": hyperboost.runhistory._n_id,
                # }
                #
                # records[task_id]["hyperboost-std"].append(hb_res)

                ########################################################################################################
                # Hyperboost EI
                ########################################################################################################

                # hyperboost = Hyperboost(scenario=scenario, rng=rng, method="EI", tae_runner=try_params)
                # hyper_start = time.time()
                # incumbent_hyperboost = hyperboost.optimize()
                # hyper_end = time.time()
                # print(f"HyperboostEI time: {hyper_end - hyper_start}")
                # hb_train, hb_test = config.get_smac_trajectories(hyperboost, model, config.NUM_ITER, X_train,
                #                                                  y_train, X_test, y_test,
                #                                                  seeds=config.SEEDS)
                # write(f"\r[HVAR] train loss = {hb_train[-1]} | test loss = {hb_test[-1]} | config = {incumbent_hyperboost._values}")
                #
                # hb_res = {
                #     "loss_train": hb_train,
                #     "loss_test": hb_test,
                #     "total_time": hyperboost.stats.wallclock_time_used,
                #     "run_time": hyperboost.stats.ta_time_used,
                #     "n_configs": hyperboost.runhistory._n_id,
                # }
                #
                # records[str_task_id][f"hyperboost-ei2"].append(hb_res)

                ########################################################################################################
                # Hyperboost QRD
                ########################################################################################################
                hyperboost = Hyperboost(scenario=scenario, rng=rng, method="QRD", tae_runner=try_params)
                hyper_start = time.time()
                incumbent_hyperboost = hyperboost.optimize()
                hyper_end = time.time()
                print(f"Hyperboost time: {hyper_end - hyper_start}")
                hb_train, hb_test = config.get_smac_trajectories(hyperboost, model, config.NUM_ITER, X_train,
                                                                 y_train, X_test, y_test,
                                                                 seeds=config.SEEDS)
                write(f"\r[HVAR] train loss = {hb_train[-1]} | test loss = {hb_test[-1]} | config = {incumbent_hyperboost._values}")

                hb_res = {
                    "loss_train": hb_train,
                    "loss_test": hb_test,
                    "total_time": hyperboost.stats.wallclock_time_used,
                    "run_time": hyperboost.stats.ta_time_used,
                    "n_configs": hyperboost.runhistory._n_id,
                }

                records[str_task_id][f"hyperboost-std-y-eps-1"].append(hb_res)

                # ########################################################################################################

                ########################################################################################################
                # Hyperboost combo
                ########################################################################################################
                # hyperboost = Hyperboost(scenario=scenario, rng=rng, method="combo", tae_runner=try_params)
                # hyper_start = time.time()
                # incumbent_hyperboost = hyperboost.optimize()
                # hyper_end = time.time()
                # print(f"Hyperboost time: {hyper_end - hyper_start}")
                # hb_train, hb_test = config.get_smac_trajectories(hyperboost, model, config.NUM_ITER, X_train,
                #                                                  y_train, X_test, y_test,
                #                                                  seeds=config.SEEDS)
                # write(f"\r[HCOM] train loss = {hb_train[-1]} | test loss = {hb_test[-1]} | config = {incumbent_hyperboost}")
                #
                # hb_res = {
                #     "loss_train": hb_train,
                #     "loss_test": hb_test,
                #     "total_time": hyperboost.stats.wallclock_time_used,
                #     "run_time": hyperboost.stats.ta_time_used,
                #     "n_configs": hyperboost.runhistory._n_id,
                # }
                #
                # records[task_id]["hyperboost-combo"].append(hb_res)

                ########################################################################################################



                ########################################################################################################

                ########################################################################################################
                # Hyperboost Drop
                ########################################################################################################
                # hyperboost = Hyperboost(scenario=scenario, rng=rng, method="drop", tae_runner=try_params)
                # hyper_start = time.time()
                # incumbent_hyperboost = hyperboost.optimize()
                # hyper_end = time.time()
                # print(f"Hyperboost time: {hyper_end - hyper_start}")
                # hb_train, hb_test = config.get_smac_trajectories(hyperboost, model, config.NUM_ITER, X_train,
                #                                                  y_train, X_test, y_test,
                #                                                  seeds=config.SEEDS)
                # write(f"\r[HDRO] train loss = {hb_train[-1]} | test loss = {hb_test[-1]} | config = {incumbent_hyperboost._values}")
                #
                # hb_res = {
                #     "loss_train": hb_train,
                #     "loss_test": hb_test,
                #     "total_time": hyperboost.stats.wallclock_time_used,
                #     "run_time": hyperboost.stats.ta_time_used,
                #     "n_configs": hyperboost.runhistory._n_id,
                # }
                #
                # records[task_id]["hyperboost-drop"].append(hb_res)

                ########################################################################################################

                ########################################################################################################
                # Hyperboost Drop DART
                ########################################################################################################
                # hyperboost = Hyperboost(scenario=scenario, rng=rng, method="drop-dart", tae_runner=try_params)
                # hyper_start = time.time()
                # incumbent_hyperboost = hyperboost.optimize()
                # hyper_end = time.time()
                # print(f"Hyperboost time: {hyper_end - hyper_start}")
                # hb_train, hb_test = config.get_smac_trajectories(hyperboost, model, config.NUM_ITER, X_train,
                #                                                  y_train, X_test, y_test,
                #                                                  seeds=config.SEEDS)
                # write(f"\r[HYBO] train loss = {hb_train[-1]} | test loss = {hb_test[-1]} | config = {incumbent_hyperboost}")
                #
                # hb_res = {
                #     "loss_train": hb_train,
                #     "loss_test": hb_test,
                #     "total_time": hyperboost.stats.wallclock_time_used,
                #     "run_time": hyperboost.stats.ta_time_used,
                #     "n_configs": hyperboost.runhistory._n_id,
                # }
                #
                # records[task_id]["hyperboost-drop-dart"].append(hb_res)

                ########################################################################################################



            write("\n")
            config.store_json(records, model.name, state)
