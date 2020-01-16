import copy
import os
from _warnings import warn
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

    def filter(self, tasks):

        # Make a copy of itself
        c = copy.copy(self)

        for gi, g in enumerate(self.groups):
            for fi, f in enumerate(g.files):

                # Filter out tasks that are not in the list
                intersection = set.intersection(set(f.tasks), set(tasks))
                c.groups[gi].files[fi].tasks = list(intersection)

                # Drop file if it does not have all tasks available
                if len(intersection) < len(tasks):
                    drop = c.groups[gi].files[fi]
                    path = os.path.join(drop.group.directory, drop.filename)
                    warn(f'{path} does not have all tasks available, dropping...')
                    c.groups[gi].files[fi] = None

            # Remove the files that were replaced with None
            c.groups[gi].files = [i for i in c.groups[gi].files if i is not None]

        return c


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
    def target_models(self):
        result = []
        for group in self.groups:
            result.append(group.target_model)
        return list(set(result))

    @property
    def overview(self):
        result = {}
        count = 0
        for group in self.groups:
            for file in group.files:
                result[count] = {
                    'Directory': group.directory,
                    'Experiment': group.prefix,
                    'Model': group.target_model,
                    'Iter. seed': file.seed,
                    'Tasks': len(file.tasks)
                }
                count += 1
        frame = pd.DataFrame(result).T
        return frame

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

    def ranker(self, seeds=None, data='train', tasks=None, include_incomplete_files=True, target_model=None):

        collected = []
        num_steps = 0

        for group in self.groups:

            # Skip group if it doesn't have the selected target model
            if group.target_model != target_model:
                continue

            # For now, we are only concerned with the target model, and nothing else.
            collected.append([file.fold_avg for file in group.files])

        num_steps = self.groups[0].files[0].array_length

        num_iterations = len(collected[0])
        task_list = self.intersection_of_tasks if include_incomplete_files else self.union_of_tasks
        task_list = task_list if not tasks else set.intersection(set(tasks), set(task_list))

        # For each iteration
        for i in range(num_iterations):
            # For each task
            for t in task_list:
                # For each step
                for s in range(num_steps):
                    # Rank across all files, skip if the group does not have enough files.
                    # `m` is for method, e.g. SMAC, Hyperboost, ROAR, etc.
                    ranks = rankdata(
                        [group[i][t][m][f'loss_{data}'][s] for group in collected if i < len(group) for m in
                         group[i][t]])

                    # And now put it back
                    counter = 0
                    for index, group in enumerate(collected):
                        if i >= len(group):
                            continue
                        for m in group[i][t]:
                            collected[index][i][t][m][f'loss_{data}'][s] = ranks[counter]
                            counter += 1

        # Put it back?
        for group_index, group in enumerate(self.groups):

            # Skip group if it doesn't have the selected target model
            if group.target_model != target_model:
                continue

            for file_index, file in enumerate(group.files):
                file.fold_rank = collected[group_index][file_index]

    def visualize(self, data='train', method='avg', tasks=None, seeds=None, include_incomplete_files=True,
                  show_std=False, target_model=None, ranked=False):
        plt.style.use('seaborn')
        # set_trace()

        if ranked:
            self.ranker(seeds=seeds, data=data, tasks=tasks, include_incomplete_files=include_incomplete_files,
                        target_model=target_model)

        # Including incomplete files, means that we use all seeds
        # If we only include complete files, we need to take the intersection of completed seeds
        if not include_incomplete_files:
            seeds = list(set.intersection(set(seeds), set(self.intersection_of_completed_seeds)))

        for group in self.groups:
            if group.target_model != target_model:
                continue
            task_mean, task_std = group.task_avg(select_tasks=tasks, include_incomplete_files=include_incomplete_files,
                                                 seeds=seeds, ranked=ranked)
            for algorithm in task_mean:
                label = group.label(algorithm)
                mean = task_mean[algorithm][f'loss_{data}'][1:]
                std = task_std[algorithm][f'loss_{data}'][1:]
                x = np.arange(start=0, stop=len(mean), step=1)
                plt.plot(mean, label=label)
                if show_std:
                    plt.fill_between(x=x, y1=mean - std, y2=mean + std, alpha=0.4)
        plt.legend()
        plt.xlabel('# Iterations')
        if ranked:
            plt.ylabel('Ranked (lower is better)')
        else:
            plt.ylabel('Loss')
        # plt.title(data)
        plt.show()
