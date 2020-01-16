import copy
from collections import defaultdict
from dataclasses import dataclass
import os
from pdb import set_trace

import numpy as np


@dataclass
class Group:
    directory: str
    prefix: str
    target_model: str
    collection: object
    files: list = None

    group_avg_: dict = None
    task_avg_: dict = None

    @property
    def id(self):
        return os.path.join(self.directory, self.prefix, self.target_model)

    def label(self, method):
        return f"{self.prefix} $\\rightarrow$ {method}"

    @property
    def intersection_of_tasks(self):
        d = [f.tasks for f in self.files]
        return set.intersection(*map(set, d))

    @property
    def union_of_tasks(self):
        d = [k for f in self.files for k in f.tasks]
        return list(set(d))

    def get_files_that_completed_tasks(self, tasks=None):
        """
        Retrieve a list of files that completed the given tasks.

        Parameters
        ----------
        tasks: list, default = self.collection.union_of_tasks
            A list of files to check.

        Returns
        -------
        files: list
            A list of files that completed the given tasks.
        """

        if tasks is None:
            tasks = self.collection.union_of_tasks

        files = []
        for f in self.files:
            if all(t in f.tasks for t in tasks):
                files.append(f)
        return files

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
                self.target_model == other.target_model
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

    @staticmethod
    def collapse_tasks(input, tasks):
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
        for task in tasks:
            for method in input[task]:
                d = input[task][method]
                aggregated[method]['loss_train'].append(np.array(d['loss_train']))
                aggregated[method]['loss_test'].append(np.array(d['loss_test']))
                aggregated[method]['total_time'].append(d['total_time'])
                aggregated[method]['run_time'].append(d['run_time'])

        return aggregated

    def subset_tasks(self, select_tasks=None, include_incomplete_files=True):
        if select_tasks:
            tasks = set(select_tasks)
        else:
            tasks = set(self.collection.union_of_tasks)

        if include_incomplete_files:
            # If we include incomplete files, we should take the intersection of tasks that are completed by the
            # incomplete files and the complete files.
            intersection_of_tasks = set(self.collection.intersection_of_tasks)
            tasks = set.intersection(tasks, intersection_of_tasks)

        else:
            # Get all tasks, but if the current file doesn't have all tasks, we should take an intersection
            # Or not: we either take the intersection of the tasks OR we exclude incomplete file.
            union_of_tasks = set(self.union_of_tasks)
            tasks = set.intersection(tasks, union_of_tasks)
        return tasks

    def subset_seeds(self, seeds=None, include_incomplete_files=True):

        if include_incomplete_files:
            files = self.files
        else:
            files = self.get_files_that_completed_tasks(self.collection.union_of_tasks)

        # Filter out files that do not have one of the included seeds
        if seeds is not None:
            files = [file for file in files if file.seed in seeds]

        return files

    def task_avg(self, select_tasks=None, include_incomplete_files=True, seeds=None, ranked=False):

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
        group_avg, group_std = self.group_avg(include_incomplete_files=include_incomplete_files, seeds=seeds, ranked=ranked)

        # Set the selected tasks, or use the union of all tasks by default
        tasks = self.subset_tasks(select_tasks=select_tasks, include_incomplete_files=include_incomplete_files)

        # Collapse group_avg and group_std
        aggregated_mean = self.collapse_tasks(group_avg, tasks)
        aggregated_var = self.collapse_tasks(group_std, tasks)

        # Calculate the mean and average std for each task
        means = defaultdict(dict)
        std = defaultdict(dict)
        for method in aggregated_mean:
            for key in aggregated_mean[method]:
                means[method][key] = np.mean(aggregated_mean[method][key], axis=0)
                std[method][key] = np.mean(np.sqrt(aggregated_var[method][key]), axis=0)

        return means, std

    def group_avg(self, include_incomplete_files=True, seeds=None, ranked=False):

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

        # Take a subset of the seeds depending on the seeds given and whether we should include incomplete files
        files = self.subset_seeds(seeds=seeds, include_incomplete_files=include_incomplete_files)

        # Collapse iterations
        aggregated = self.collapse_iterations(files, ranked=ranked)

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
