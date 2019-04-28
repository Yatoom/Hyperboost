import time

import numpy as np
import openml
from sklearn.model_selection import KFold
from smac.facade.smac_facade import SMAC

from benchmarks import config
from benchmarks.config import create_smac_runner
from benchmarks.param_spaces import RandomForestSpace, DecisionTreeSpace, SVMSpace
from benchmarks.preprocessing import ConditionalImputer
from hyperboost.hyperboost import Hyperboost


def write(*args, **kwargs):
    with open("output.txt", "a+") as f:
        f.write(*args)


for state in config.SEEDS:
    rng = np.random.RandomState(state)
    for model in [SVMSpace, RandomForestSpace]:
        records = {}
        for task_id in config.TASKS:

            records[task_id] = {"smac": [], "hyperboost-qr": [], "hyperboost-qrd": [], "hyperboost-drop": []}

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

                # ########################################################################################################
                # # SMAC
                # ########################################################################################################
                # smac = SMAC(scenario=scenario, rng=rng, tae_runner=try_params, use_pynisher=False)
                # smac_start = time.time()
                # incumbent_smac = smac.optimize()
                # smac_end = time.time()
                # print(f"SMAC time: {smac_end - smac_start}")
                # smac_train, smac_test = config.get_smac_trajectories(smac, model, config.NUM_ITER, X_train,
                #                                                      y_train, X_test, y_test,
                #                                                      seeds=config.SEEDS)
                # write(f"\r[SMAC] train loss = {smac_train[-1]} | test loss = {smac_test[-1]}")
                # smac_res = {
                #     "loss_train": smac_train,
                #     "loss_test": smac_test,
                #     "total_time": smac.stats.wallclock_time_used,
                #     "run_time": smac.stats.ta_time_used,
                #     "n_configs": smac.runhistory._n_id,
                # }
                # records[task_id]["smac"].append(smac_res)
                #
                # ########################################################################################################

                # ########################################################################################################
                # # Hyperboost QR
                # ########################################################################################################
                # hyperboost = Hyperboost(scenario=scenario, rng=rng, method="QR", tae_runner=try_params)
                # hyper_start = time.time()
                # incumbent_hyperboost = hyperboost.optimize()
                # hyper_end = time.time()
                # print(f"Hyperboost time: {hyper_end - hyper_start}")
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
                # records[task_id]["hyperboost-qr"].append(hb_res)
                #
                # ########################################################################################################

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
                write(f"\r[HYBO] train loss = {hb_train[-1]} | test loss = {hb_test[-1]}")

                hb_res = {
                    "loss_train": hb_train,
                    "loss_test": hb_test,
                    "total_time": hyperboost.stats.wallclock_time_used,
                    "run_time": hyperboost.stats.ta_time_used,
                    "n_configs": hyperboost.runhistory._n_id,
                }

                records[task_id]["hyperboost-qrd"].append(hb_res)

                ########################################################################################################

                # ########################################################################################################
                # # Hyperboost Drop
                # ########################################################################################################
                # hyperboost = Hyperboost(scenario=scenario, rng=rng, method="drop", tae_runner=try_params)
                # hyper_start = time.time()
                # incumbent_hyperboost = hyperboost.optimize()
                # hyper_end = time.time()
                # print(f"Hyperboost time: {hyper_end - hyper_start}")
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
                # records[task_id]["hyperboost-drop"].append(hb_res)
                #
                # ########################################################################################################



            write("\n")
            config.store_json(records, model.name, state)
