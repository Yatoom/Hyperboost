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

        group_avg = self.group_avg(include_incomplete_files=include_incomplete_files, seeds=seeds)
        result = defaultdict(lambda: defaultdict(lambda: np.zeros(self.array_length)))

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

        for task in tasks:
            count_tasks += 1
            for method in group_avg[task]:
                d = group_avg[task][method]
                result[method]['loss_train'] += np.array(d['loss_train']) / num_tasks
                result[method]['loss_test'] += np.array(d['loss_test']) / num_tasks
                result[method]['total_time'] += d['total_time'] / num_tasks
                result[method]['run_time'] += d['run_time'] / num_tasks
                result[method]['n_configs'] += d['n_configs'] / num_tasks

        assert(num_tasks == count_tasks)
        return result

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
                    lambda: np.zeros(array_length, )
                )
            )
        )

        for file in files:
            for task in file.fold_avg:
                for method in file.data[task]:
                    d = file.fold_avg[task][method]
                    aggregated[task][method]['loss_train'] += np.array(d['loss_train']) / num_files
                    aggregated[task][method]['loss_test'] += np.array(d['loss_test']) / num_files
                    aggregated[task][method]['total_time'] += d['total_time'] / num_files
                    aggregated[task][method]['run_time'] += d['run_time'] / num_files
                    aggregated[task][method]['n_configs'] += d['n_configs'] / num_files

        self.group_avg_ = aggregated
        return aggregated
