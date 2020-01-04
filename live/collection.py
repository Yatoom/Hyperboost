import os

from live import File, Group
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
        d = [k for g in self.groups for k in list(g.common_tasks)]
        return list(set(d))

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

    def visualize_streamlit(self, data='train', method='avg', tasks=None):
        frame = pd.DataFrame()
        for group in self.groups:
            task_avg = group.task_avg(select_tasks=tasks)
            for algorithm in task_avg:
                label = f"{group.prefix}-{group.target_model}-{algorithm}"
                frame[label] = task_avg[algorithm][f'loss_{data}'][1:]
        # return frame)
        print('iplot')
        fig = frame.iplot(kind='line')
        # fig = px.line(frame)
        print('st plotly')
        st.plotly_chart(fig)


    def visualize(self, data='train', method='avg', tasks=None):
        plt.style.use('seaborn')
        for group in self.groups:
            task_avg = group.task_avg(select_tasks=tasks)
            for algorithm in task_avg:
                label = f"{group.prefix}-{group.target_model}-{algorithm}"
                plt.plot(task_avg[algorithm][f'loss_{data}'][1:], label=label)
        plt.legend()
        plt.xlabel('# Iterations')
        plt.ylabel('Loss')
        # plt.title(data)
        plt.show()
