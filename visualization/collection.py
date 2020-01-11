import os
from collections import defaultdict
from pdb import set_trace

from visualization import File, Group
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
import cufflinks as cf


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
        seeds = [file.seed for group in self.groups for file in group.get_files_that_completed_tasks(self.union_of_tasks)]
        return list(set(seeds))

    @property
    def intersection_of_completed_seeds(self):
        seeds = [set(file.seed for file in group.get_files_that_completed_tasks(self.union_of_tasks)) for group in self.groups]
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

    # def visualize_streamlit(self, data='train', method='avg', tasks=None):
    #     frame = pd.DataFrame()
    #     for group in self.groups:
    #         task_avg = group.task_avg(select_tasks=tasks)
    #         for algorithm in task_avg:
    #             label = f"{group.prefix}-{group.target_model}-{algorithm}"
    #             frame[label] = task_avg[algorithm][f'loss_{data}'][1:]
    #     # return frame)
    #     print('iplot')
    #     fig = frame.iplot(kind='line')
    #     # fig = px.line(frame)
    #     print('st plotly')
    #     st.plotly_chart(fig)

    def visualize(self, data='train', method='avg', tasks=None, seeds=None, include_incomplete_files=True):
        plt.style.use('seaborn')
        # set_trace()

        # Including incomplete files, means that we use all seeds
        # If we only include complete files, we need to take the intersection of completed seeds
        if not include_incomplete_files:
            seeds = list(set.intersection(set(seeds), set(self.intersection_of_completed_seeds)))

        print(tasks, seeds, include_incomplete_files)

        for group in self.groups:
            task_avg = group.task_avg(select_tasks=tasks, include_incomplete_files=include_incomplete_files, seeds=seeds)
            for algorithm in task_avg:
                label = f"{group.prefix}-{group.target_model}-{algorithm}"
                plt.plot(task_avg[algorithm][f'loss_{data}'][1:], label=label)
        plt.legend()
        plt.xlabel('# Iterations')
        plt.ylabel('Loss')
        # plt.title(data)
        plt.show()
