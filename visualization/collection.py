import copy
import os
import pathlib
from _warnings import warn
import plotly.graph_objects as go

from visualization import File, Group
import pandas as pd
import streamlit as st
import numpy as np
from scipy.stats import rankdata


class Collection:
    def __init__(self):
        self.groups = []

        self.colors = [
            'rgba(31, 119, 180, {})',  # muted blue
            'rgba(255, 127, 14, {})',  # safety orange
            'rgba(44, 160, 44, {})',  # cooked asparagus green
            'rgba(214, 39, 40, {})',  # brick red
            'rgba(148, 103, 189, {})',  # muted purple
            'rgba(140, 86, 75, {})',  # chestnut brown
            'rgba(227, 119, 194, {})',  # raspberry yogurt pink
            'rgba(127, 127, 127, {})',  # middle gray
            'rgba(188, 189, 34, {})',  # curry yellow-green
            'rgba(23, 190, 207, {})'  # blue-teal
        ]

    def get_color(self, num, alpha):
        num = num % len(self.colors)
        return self.colors[num].format(alpha)

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

        c.groups = [group for group in c.groups if len(group.files) > 0]

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
                    'Target': group.target_name,
                    'Iter. seed': file.seed,
                    'Tasks': len(file.tasks)
                }
                count += 1
        frame = pd.DataFrame(result).T
        return frame

    def add_files(self, directory):
        for filename in os.listdir(directory):

            # Gather information from filename
            dirname = pathlib.PurePath(directory).name
            names = filename.replace('.json', '').split('-')

            target_name = names[-2]
            seed = names[-1]
            prefix = dirname + "/" + '-'.join(names[0:-2])

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
                        for m in group[i_capped][t]:
                            result[index][i_capped][t][m][f'loss_{data}'][s] = ranks[counter]
                            counter += 1

        # Put it back?
        for group_index, group in enumerate(self.groups):

            for file_index, file in enumerate(group.files):
                file.fold_rank = result[group_index][file_index]

        return self

    def get_wins(self, baseline):
        get_trajectories = lambda files: pd.concat([f.agg_trajectories() for f in files], axis=1)
        collected = [get_trajectories(group.files).rename(lambda x: f"{group.prefix}-{x}", axis=1) for group in self.groups]
        concatenated = pd.concat(collected, axis=1).T.groupby(level=0).agg(list).T
        aggregated = concatenated.applymap(lambda x: np.median(x, axis=0))
        compared = aggregated.divide(aggregated[baseline], axis=0).applymap(lambda x: x <= 1)
        result = compared.T.agg(list, axis=1).map(lambda x: np.mean(x, axis=0))
        return pd.DataFrame(result.to_dict())[1:]

    def result_table(self, compare_with_col, seed_agg=np.median, outer_loop_agg=np.mean):

        # Create a table of the latest scores
        get_bests = lambda files: pd.concat([f.get_bests(outer_loop_agg) for f in files], axis=1).T.groupby(level=0).agg(seed_agg).T

        # Create tables for all groups and concatenate them with good columnn names
        collected = pd.concat([get_bests(group.files).rename(lambda x: f"{group.prefix}-{x}", axis=1) for group in self.groups], axis=1)

        result = (collected.divide(collected[compare_with_col], axis=0) <= 1)
        result = pd.DataFrame(result.mean(axis=0), columns=[compare_with_col])

        return result


    def visualize_wins(self, baseline):
        wins = self.get_wins(baseline)
        f = go.FigureWidget()

        color_counter = 0
        for win in wins:
            solid_color = self.get_color(color_counter, 1)
            color_counter += 1
            f.add_scattergl(y=list(wins[win]), name=win, line_color=solid_color)
        return f

    def visualize(self, data='train', show_std=False, ranked=False, global_std=False):
        f = go.FigureWidget()
        color_counter = 0
        if ranked:
            self.calculate_ranks(data=data)

        # Additional table
        table = {}

        # Including incomplete files, means that we use all seeds
        # If we only include complete files, we need to take the intersection of completed seeds
        for group in self.groups:
            task_mean, task_std = group.task_avg(ranked=ranked, global_std=global_std)

            for algorithm in task_mean:
                label = group.label(algorithm)
                mean = task_mean[algorithm][f'loss_{data}'][1:]
                std = task_std[algorithm][f'loss_{data}'][1:]
                x = np.arange(start=0, stop=len(mean), step=1)

                # Write to table
                table[label] = {
                    'avg. total runtime': f"{task_mean[algorithm]['total_time'] : .2f} ± {task_std[algorithm]['total_time'] :.2f}",
                    'avg. target eval. time': f"{task_mean[algorithm]['run_time'] : .2f} ± {task_std[algorithm]['run_time'] :.2f}",
                }

                # Plotting
                solid_color = self.get_color(color_counter, 1)
                transparent_color = self.get_color(color_counter, 0.5)
                color_counter += 1
                f.add_scattergl(y=list(mean), name=label, line_color=solid_color)

                if show_std:
                    scat1 = go.Scatter(
                        y=mean - std,
                        name=label,
                        # fillcolor=transparent_color,
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False)
                    scat2 = go.Scatter(
                        y=mean + std,
                        name=label,
                        fill='tonexty',
                        fillcolor=transparent_color,
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False)
                    f.add_trace(scat1)
                    f.add_trace(scat2)

        f.update_layout(
            title=f"Average {'ranking' if ranked else 'loss'} on {data} data ",
            xaxis_title="Iteration",
            yaxis_title="Ranked (lower is better)" if ranked else "Loss",
        )

        return f, pd.DataFrame(table).T
        # st.plotly_chart(f)
        # st.table()
