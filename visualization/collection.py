import os
from collections import defaultdict

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
    def common_tasks(self):
        d = [g.common_tasks for g in self.groups]
        return list(set.intersection(*map(set, d)))

    @property
    def all_tasks(self):
        # FIXME: I think we should take the intersection of all_tasks of each group
        d = [k for g in self.groups for k in list(g.common_tasks)]
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
    def all_seeds(self):
        seeds = [file.seed for group in self.groups for file in group.complete_files]
        return list(set(seeds))

    @property
    def completed_seeds(self):
        seeds = [set(file.seed for file in group.complete_files) for group in self.groups]
        return list(set.intersection(*seeds))


    def add_files(self, directory):
        for filename in os.listdir(directory):
            # Gather information from filename
            prefix, target_model, seed = filename.replace('.json', '').split('-')

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
        seeds = list(set.intersection(set(seeds), set(self.completed_seeds))) if include_incomplete_files else seeds

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