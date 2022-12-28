import copy
import json
import os
from typing import Union

import numpy as np

from visualization.group import Group


class File:
    def __init__(self, filename: str, group: Group, seed: Union[int, str]):
        self.filename = filename
        self.group = group
        self.seed = int(seed)
        self.data, self.tasks = self.remove_incomplete_runs(self.load_data())

        self.data_ = None
        self.fold_avg_ = None

    def load_data(self):
        path = os.path.join(self.group.directory, self.filename)
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    @staticmethod
    def remove_incomplete_runs(file_data):
        """
        Removes incomplete runs from the JSON file.

        Parameters
        ----------
        file_data: JSON data
            The raw JSON data

        Returns
        -------
        result: dict
            The cleaned data

        included_tasks: list
            The list of included tasks in the result
        """

        result = copy.copy(file_data)

        # Task number, e.g. 49, 31, etc.
        for task in file_data.keys():

            # Method, e.g. hyperboost, smac, etc.
            for method in file_data[task].keys():

                # Remove method if it doesn't contain runs
                if len(file_data[task][method]) == 0:
                    del result[task][method]

            # Remove task number, if it doesn't contain runs
            if len(file_data[task].keys()) == 0:
                del file_data[task]

        included_tasks = list(result.keys())

        return result, included_tasks

    @property
    def array_length(self):
        task = self.first_key(self.data)
        method = self.first_key(self.data[task])
        fold = self.data[task][method][0]

        return len(fold['loss_train'])

    @staticmethod
    def first_key(dictionary):
        return list(dictionary.keys())[0]

    @property
    def fold_avg(self):

        # INPUT:
        #
        # task > method > fold > loss_train
        #                      > loss_test
        #                      > total_time
        #                      > run_time
        #                      > n_configs
        #
        # OUTPUT:
        #
        # task > method > avg. loss_train
        #               > avg. loss_test
        #               > avg. total_time
        #               > avg. run_time
        #               > avg. n_configs

        # Return cached data
        if self.fold_avg_:
            return self.fold_avg_

        # Calculate result
        result = {}

        for task in self.data:
            result[task] = {}
            for method in self.data[task]:
                result[task][method] = {
                    "loss_train": np.mean([i['loss_train'] for i in self.data[task][method]], axis=0),
                    "loss_test": np.mean([i['loss_test'] for i in self.data[task][method]], axis=0),
                    "total_time": np.mean([i['total_time'] for i in self.data[task][method]]),
                    "run_time": np.mean([i['run_time'] for i in self.data[task][method]]),
                    "n_configs": np.mean([i['n_configs'] for i in self.data[task][method]]),
                }

        self.fold_avg_ = result

        return self.fold_avg_
