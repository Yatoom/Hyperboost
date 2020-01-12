import os
from collections import defaultdict
from pdb import set_trace

from visualization import File, Group
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
import cufflinks as cf
import numpy as np
from scipy.stats import rankdata


class Collection:
    def __init__(self):
        self.groups = []

    @property
    def intersection_of_tasks(self):
        d = [g.intersection_of_tasks for g in self.groups]
        return list(set.intersection(*map(set, d)))

    @property
    def union_of_tasks(self):
        d = [k for g in self.groups for k in list(g.union_of_tasks)]
        return list(set(d))

    @property
    def union_of_intersections_of_tasks(self):
        d = [k for g in self.groups for k in list(g.intersection_of_tasks)]
        return list(set(d))

    @property
    def overview(self):
        result = {}
        count = 0
        for group in self.groups:
            for file in group.files:
                result[count] = {
                    'Experiment': group.id,
                    'Iteration seed': file.seed,
                    'Tasks completed': len(file.tasks)
                }
                count += 1
        return pd.DataFrame(result).T

    @property
    def union_of_completed_seeds(self):
        seeds = [file.seed for group in self.groups for file in
                 group.get_files_that_completed_tasks(self.union_of_tasks)]
        return list(set(seeds))

    @property
    def intersection_of_completed_seeds(self):
        seeds = [set(file.seed for file in group.get_files_that_completed_tasks(self.union_of_tasks)) for group in
                 self.groups]
        return list(set.intersection(*seeds))

    @property
    def intersection_of_any_seeds(self):
        # Same as completed seeds, except that they don't have to be complete.
        seeds = [set(file.seed for file in group.files) for group in self.groups]
        return list(set.intersection(*seeds))

    def add_files(self, directory):
        for filename in os.listdir(directory):
            # Gather information from filename
            names = filename.replace('.json', '').split('-')
            target_model = names[-2]
            seed = names[-1]
            prefix = '-'.join(names[0:-2])

            # print(prefix, target_model, seed)

            # Create or get group
            group = self.get_group(
                directory=directory,
                prefix=prefix,
                target_model=target_model,
                collection=self
            )

            # Create file object
            file = File(
                filename=filename,
                seed=seed,
                group=group
            )

            # Add file to group
            group.files.append(file)

        return self

    def get_group(self, *args, **kwargs):
        group = Group(*args, **kwargs)

        for existing_group in self.groups:
            if existing_group == group:
                return existing_group

        self.groups.append(group)
        group.files = []
        return group

    def combine_key_method_values(self, select_tasks=None, include_incomplete_files=True, seeds=None):
        combined = defaultdict(dict)
        for group in self.groups:
            r = group.get_key_method_values(select_tasks=select_tasks,
                                            include_incomplete_files=include_incomplete_files, seeds=seeds)
            for key in r:
                for method in r[key]:
                    combined[key][method] = r[key][method]
        return combined

    def rank(self, data='train', tasks=None, include_incomplete_files=True, seeds=None, show_std=True):
        key = f'loss_{data}'

        combined = self.combine_key_method_values(select_tasks=tasks,
                                                  include_incomplete_files=include_incomplete_files, seeds=seeds)

        first_key = list(combined[key].keys())[0]
        num_iterations =len(combined[key][first_key][0])
        num_tasks = len(combined[key][first_key])

        ranked = defaultdict(
                lambda: defaultdict(
                    list
                )
        )
        # We need to make a separate ranking for every task for every step
        for iteration in range(num_iterations):
            for task in range(num_tasks):
                data = [combined[key][method][task][iteration] for method in combined[key]]
                ranked_data = rankdata(data)
                for i, method in enumerate(combined[key]):
                    ranked[method][task].append(ranked_data[i])

        # Take average
        mean_rank = dict()
        std_rank = dict()
        plt.style.use('seaborn')
        for method in ranked:

            mean_rank[method] = np.mean(list(ranked[method].values()), axis=0)
            std_rank[method] = np.std(list(ranked[method].values()), axis=0)

            plt.plot(mean_rank[method], label=method)
            if show_std:
                x = np.arange(start=0, stop=len(mean_rank[method]), step=1)
                plt.fill_between(x, mean_rank[method] - std_rank[method], mean_rank[method] + std_rank[method], alpha=0.2)

        plt.legend()
        plt.xlabel('# Iterations')
        plt.ylabel('Loss')
        # plt.title(data)
        plt.show()



    def visualize(self, data='train', method='avg', tasks=None, seeds=None, include_incomplete_files=True,
                  show_std=False):
        plt.style.use('seaborn')
        # set_trace()

        # Including incomplete files, means that we use all seeds
        # If we only include complete files, we need to take the intersection of completed seeds
        if not include_incomplete_files:
            seeds = list(set.intersection(set(seeds), set(self.intersection_of_completed_seeds)))

        for group in self.groups:
            task_mean, task_std = group.task_avg(select_tasks=tasks, include_incomplete_files=include_incomplete_files,
                                                 seeds=seeds)
            for algorithm in task_mean:
                label = f"{group.prefix}-{group.target_model}-{algorithm}"
                mean = task_mean[algorithm][f'loss_{data}'][1:]
                std = task_std[algorithm][f'loss_{data}'][1:]
                x = np.arange(start=0, stop=len(mean), step=1)
                plt.plot(mean, label=label)
                if show_std:
                    plt.fill_between(x=x, y1=mean - std, y2=mean + std, alpha=0.4)
        plt.legend()
        plt.xlabel('# Iterations')
        plt.ylabel('Loss')
        # plt.title(data)
        plt.show()
