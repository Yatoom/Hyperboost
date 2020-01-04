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
        return os.path.join(self.directory, self.prefix, self.target_model, self.files)

    @property
    def common_tasks(self):
        d = [f.tasks for f in self.files]
        return set.intersection(*map(set, d))

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

    def task_avg(self, select_tasks=None):
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

        if self.task_avg_:
            return self.task_avg_

        group_avg = self.group_avg
        num_tasks = len(group_avg.keys())
        result = defaultdict(lambda: defaultdict(lambda: np.zeros(self.array_length)))

        tasks = self.collection.common_tasks
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

        self.task_avg_ = result
        return result

    @property
    def group_avg(self):

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

        if self.group_avg_:
            return self.group_avg_

        array_length = self.array_length
        num_files = len(self.files)

        aggregated = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: np.zeros(array_length, )
                )
            )
        )

        for file in self.files:
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
