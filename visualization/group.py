from collections import defaultdict
from dataclasses import dataclass
import os

import numpy as np


@dataclass
class Group:
    directory: str
    prefix: str
    target_name: str
    collection: object
    files: list = None

    group_avg_: dict = None
    task_avg_: dict = None

    @property
    def id(self):
        return os.path.join(self.directory, self.prefix, self.target_name)

    def label(self, method):
        return f"{self.prefix} > {method}"

    @property
    def intersection_of_tasks(self):
        d = [f.tasks for f in self.files]
        return set.intersection(*map(set, d))

    @property
    def union_of_tasks(self):
        d = [k for f in self.files for k in f.tasks]
        return list(set(d))

    @property
    def union_of_seeds(self):
        return [file.seed for file in self.files]

    @property
    def array_length(self):
        return self.files[0].array_length

    def __eq__(self, other):
        return (
                self.directory == other.directory and
                self.prefix == other.prefix and
                self.target_name == other.target_name
        )

    @staticmethod
    def collapse_iterations(files, ranked=False):
        # INPUT:
        #
        # file > task > method > avg. loss_train
        #                      > avg. loss_test
        #                      > avg. total_time
        #                      > avg. run_time
        #                      > avg. n_configs
        #
        # OUTPUT:
        #
        # task > method > [loss_train, loss_train, ...]
        #               > [loss_test, loss_test, ...]
        #               > [total_time, total_time, ...]
        #               > [run_time, run_time, ...]
        #               > [n_configs, n_configs, ...]

        aggregated = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: []
                )
            )
        )

        for file in files:
            target = file.fold_rank if ranked else file.fold_avg
            for task in target:
                for method in file.data[task]:
                    d = target[task][method]
                    aggregated[task][method]['loss_train'].append(np.array(d['loss_train']))
                    aggregated[task][method]['loss_test'].append(np.array(d['loss_test']))
                    aggregated[task][method]['total_time'].append(d['total_time'])
                    aggregated[task][method]['run_time'].append(d['run_time'])
                    aggregated[task][method]['n_configs'].append(d['n_configs'])

        return aggregated

    def collapse_tasks(self, input):
        # INPUT:
        #
        # task > method > avg. loss_train
        #               > avg. loss_test
        #               > avg. total_time
        #               > avg. run_time
        #               > avg. n_configs
        #
        # OUTPUT:
        #
        # method > [loss_train, loss_train, ...]
        #        > [loss_test, loss_test, ...]
        #        > [total_time, total_time, ...]
        #        > [run_time, run_time, ...]
        #        > [n_configs, n_configs, ...]

        aggregated = defaultdict(lambda: defaultdict(lambda: []))
        for task in self.collection.tasks:
            for method in input[task]:
                d = input[task][method]
                aggregated[method]['loss_train'].append(np.array(d['loss_train']))
                aggregated[method]['loss_test'].append(np.array(d['loss_test']))
                aggregated[method]['total_time'].append(d['total_time'])
                aggregated[method]['run_time'].append(d['run_time'])

        return aggregated

    def task_avg(self, ranked=False):

        # INPUT: (group_avg, group_std)
        #
        # task > method > avg. loss_train
        #               > avg. loss_test
        #               > avg. total_time
        #               > avg. run_time
        #               > avg. n_configs
        #
        # OUTPUT: (means, std)
        #
        # method > avg. loss_train
        #        > avg. loss_test
        #        > avg. total_time
        #        > avg. run_time
        #        > avg. n_configs

        # Get average and standard deviation over iterations
        group_avg, group_std = self.group_avg(ranked=ranked)

        # Set the selected tasks, or use the union of all tasks by default
        # Collapse group_avg and group_std
        aggregated_mean = self.collapse_tasks(group_avg)
        aggregated_var = self.collapse_tasks(group_std)

        # Calculate the mean and average std for each task
        means = defaultdict(dict)
        std = defaultdict(dict)
        for method in aggregated_mean:
            for key in aggregated_mean[method]:
                means[method][key] = np.mean(aggregated_mean[method][key], axis=0)
                std[method][key] = np.mean(np.sqrt(aggregated_var[method][key]), axis=0)

        return means, std

    def group_avg(self, ranked=False):

        # INPUT:
        #
        # file > task > method > avg. loss_train
        #                      > avg. loss_test
        #                      > avg. total_time
        #                      > avg. run_time
        #                      > avg. n_configs
        #
        # OUTPUT:
        #
        # task > method > avg. loss_train
        #               > avg. loss_test
        #               > avg. total_time
        #               > avg. run_time
        #               > avg. n_configs

        # Collapse iterations
        aggregated = self.collapse_iterations(self.files, ranked=ranked)

        # Setup default dictionaries for means and vars
        means = defaultdict(lambda: defaultdict(dict))
        var = defaultdict(lambda: defaultdict(dict))

        # Compute means and vars
        for task in aggregated:
            for method in aggregated[task]:
                for key in aggregated[task][method]:
                    means[task][method][key] = np.mean(aggregated[task][method][key], axis=0)
                    var[task][method][key] = np.var(aggregated[task][method][key], axis=0)

        return means, var
