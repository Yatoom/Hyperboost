import copy
from collections import defaultdict
from dataclasses import dataclass
import os
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
    def common_tasks(self):
        d = [f.tasks for f in self.files]
        return set.intersection(*map(set, d))

    @property
    def complete_files(self):
        num_tasks = len(self.collection.all_tasks)
        files = []
        for f in self.files:
            if len(list(f.tasks)) == num_tasks:
                files.append(f)
        return files

    @property
    def all_tasks(self):
        d = [k for f in self.files for k in f.tasks]
        return list(set(d))

    @property
    def all_seeds(self):
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

        tasks = self.collection.common_tasks if include_incomplete_files else self.all_tasks
        num_tasks = len(tasks)

        if select_tasks:
            tasks = select_tasks

        for task in tasks:
            for method in group_avg[task]:
                d = group_avg[task][method]
                result[method]['loss_train'] += np.array(d['loss_train']) / num_tasks
                result[method]['loss_test'] += np.array(d['loss_test']) / num_tasks
                result[method]['total_time'] += d['total_time'] / num_tasks
                result[method]['run_time'] += d['run_time'] / num_tasks
                result[method]['n_configs'] += d['n_configs'] / num_tasks

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
        files = self.files if include_incomplete_files else self.complete_files

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

        print(num_files)

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
