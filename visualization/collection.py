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

    def filter(self, tasks, target_name=None):

        # Make a copy of itself
        c = copy.copy(self)

        for gi, g in enumerate(self.groups):
            for fi, f in enumerate(g.files):

                # Drop group if it doesn't have the selected target model
                if target_name is not None and g.target_name != target_name:
                    del c.groups[gi]
                    continue

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
    def tasks(self):
        return set([task for group in self.groups for task in group.union_of_tasks])

    @property
    def common_tasks(self):
        return set.intersection(*[set(group.intersection_of_tasks) for group in self.groups])

    @property
    def targets(self):
        target_names = [group.target_name for group in self.groups]
        return list(set(target_names))

    @property
    def overview(self):
        result = {}
        count = 0
        for group in self.groups:
            for file in group.files:
                result[count] = {
                    'Directory': group.directory,
                    'Experiment': group.prefix,
                    'Model': group.target_name,
                    'Iter. seed': file.seed,
                    'Tasks': len(file.tasks)
                }
                count += 1
        frame = pd.DataFrame(result).T
        return frame

    def add_files(self, directory):
        for filename in os.listdir(directory):

            # Gather information from filename
            names = filename.replace('.json', '').split('-')
            target_name = names[-2]
            seed = names[-1]
            prefix = '-'.join(names[0:-2])

            # Create or get group
            group = self.get_group(
                directory=directory,
                prefix=prefix,
                target_name=target_name,
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

    def calculate_ranks(self, data='train'):

        collected = [[file.fold_avg for file in group.files] for group in self.groups]
        result = copy.deepcopy(collected)
        num_steps = self.groups[0].files[0].array_length
        num_iterations = len(collected[0])

        print('num_iterations', num_iterations)

        # For each iteration
        for i in range(num_iterations):
            # For each task
            for t in self.tasks:
                # For each step
                for s in range(num_steps):

                    # Rank across all files, cap i if the group does not have enough files.
                    array_for_ranking = []
                    for group in collected:
                        i_capped = min(i, len(group) - 1)
                        for method in group[i_capped][t]:
                            d = group[i_capped][t][method][f'loss_{data}'][s]
                            array_for_ranking.append(d)

                    ranks = rankdata(array_for_ranking)

                    # And now put it back
                    counter = 0
                    for index, group in enumerate(collected):
                        i_capped = min(i, len(group) - 1)
                        # if i >= len(group):
                        #     continue
                        for m in group[i_capped][t]:
                            result[index][i_capped][t][m][f'loss_{data}'][s] = ranks[counter]
                            counter += 1

        # Put it back?
        for group_index, group in enumerate(self.groups):

            for file_index, file in enumerate(group.files):
                file.fold_rank = result[group_index][file_index]

        return self

    def visualize(self, data='train', show_std=False, ranked=False):
        plt.style.use('seaborn')

        if ranked:
            self.calculate_ranks(data=data)

        # Including incomplete files, means that we use all seeds
        # If we only include complete files, we need to take the intersection of completed seeds
        for group in self.groups:
            task_mean, task_std = group.task_avg(ranked=ranked)

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
        plt.show()
