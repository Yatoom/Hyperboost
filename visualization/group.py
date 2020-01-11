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

    @property
    def task_rank(self):
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
        # method > ranked loss_train
        #        > ranked loss_test
        #        > ranked total_time
        #        > ranked run_time
        #        > ranked n_configs

        raise NotImplementedError()

    def task_avg(self, select_tasks=None, include_incomplete_files=True, seeds=None):
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
        # method > avg. loss_train
        #        > avg. loss_test
        #        > avg. total_time
        #        > avg. run_time
        #        > avg. n_configs

        group_avg, group_std = self.group_avg(include_incomplete_files=include_incomplete_files, seeds=seeds)

        # Set the selected tasks, or use the union of all tasks by default
        if select_tasks:
            tasks = set(select_tasks)
        else:
            tasks = set(self.collection.union_of_tasks)

        if include_incomplete_files:
            # If we include incomplete files, we should take the intersection of tasks that are completed by the
            # incomplete files and the complete files.

            # FIXME: we should not get
            intersection_of_tasks = set(self.collection.intersection_of_tasks)
            tasks = set.intersection(tasks, intersection_of_tasks)

        else:
            # Get all tasks, but if the current file doesn't have all tasks, we should take an intersection
            # Or not: we either take the intersection of the tasks OR we exclude incomplete file.
            union_of_tasks = set(self.union_of_tasks)
            tasks = set.intersection(tasks, union_of_tasks)

        num_tasks = len(tasks)
        count_tasks = 0

        aggregated_mean = defaultdict(lambda: defaultdict(lambda: []))
        aggregated_var = defaultdict(lambda: defaultdict(lambda: []))

        for task in tasks:
            count_tasks += 1
            for method in group_avg[task]:
                d = group_avg[task][method]
                aggregated_mean[method]['loss_train'].append(np.array(d['loss_train']))
                aggregated_mean[method]['loss_test'].append(np.array(d['loss_test']))
                aggregated_mean[method]['total_time'].append(d['total_time'])
                aggregated_mean[method]['run_time'].append(d['run_time'])
                aggregated_mean[method]['n_configs'].append(d['n_configs'])

        for task in tasks:
            count_tasks += 1
            for method in group_std[task]:
                d = group_std[task][method]
                aggregated_var[method]['loss_train'].append(np.array(d['loss_train']))
                aggregated_var[method]['loss_test'].append(np.array(d['loss_test']))
                aggregated_var[method]['total_time'].append(d['total_time'])
                aggregated_var[method]['run_time'].append(d['run_time'])
                aggregated_var[method]['n_configs'].append(d['n_configs'])

        means = defaultdict(dict)
        std = defaultdict(dict)
        for method in aggregated_mean:
            for key in aggregated_mean[method]:
                means[method][key] = np.mean(aggregated_mean[method][key], axis=0)
                std[method][key] = np.mean(np.sqrt(aggregated_var[method][key]), axis=0)

                # std[method][key] = np.array(

                    # Law of total variance: mean of variance plus variance of mean
                    # np.mean(aggregated_var[method][key], axis=0) + np.var(aggregated_mean[method][key], axis=0)
                # )

        assert (num_tasks * 2 == count_tasks)
        return means, std

    def group_avg(self, include_incomplete_files=True, seeds=None):

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

        array_length = self.array_length

        if include_incomplete_files:
            files = self.files
        else:
            files = self.get_files_that_completed_tasks(self.collection.union_of_tasks)

        # Filter out files that do not have one of the included seeds
        if seeds is not None:
            files = [file for file in files if file.seed in seeds]

        num_files = len(files)

        aggregated = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: []
                )
            )
        )

        for file in files:
            for task in file.fold_avg:
                for method in file.data[task]:
                    d = file.fold_avg[task][method]
                    aggregated[task][method]['loss_train'].append(np.array(d['loss_train']))
                    aggregated[task][method]['loss_test'].append(np.array(d['loss_test']))
                    aggregated[task][method]['total_time'].append(d['total_time'])
                    aggregated[task][method]['run_time'].append(d['run_time'])
                    aggregated[task][method]['n_configs'].append(d['n_configs'])

        means = defaultdict(
            lambda: defaultdict(dict)
        )

        var = defaultdict(
            lambda: defaultdict(dict)
        )

        for task in aggregated:
            for method in aggregated[task]:
                for key in aggregated[task][method]:
                    means[task][method][key] = np.mean(aggregated[task][method][key], axis=0)
                    var[task][method][key] = np.var(aggregated[task][method][key], axis=0)

        return means, var
