import os

import streamlit as st
import pandas as pd

from benchmark.config import TASKS, SEEDS
from visualization import Collection, variables

c = Collection()
c = c.add_files('baseline/')
c = c.add_files('output/results4/')

num_iterations = len(os.listdir('output/results3/'))
num_seeds = len(SEEDS)

st.header('Settings')
# Show overview of the data we have available
st.subheader('Collection')
st.table(c.overview)

# Make a subset of the data
st.subheader('Selection')
subset_type = st.radio(
    'Choose a subset to visualize in order to make the comparison between an existing and a running experiment fair.',
    (variables.REMOVE_INCOMPLETE, variables.TASK_INTERSECTION))
subset_iterations = st.radio(
    'From the subset above, for each experiment:',
    (variables.ALL_SEEDS, variables.SEED_INTERSECTION)
)

# Set the selection
tasks = c.intersection_of_tasks if subset_type == variables.TASK_INTERSECTION else c.union_of_tasks
include_incomplete_files = subset_type != variables.REMOVE_INCOMPLETE
seeds = c.union_of_completed_seeds if subset_iterations == variables.ALL_SEEDS else c.intersection_of_completed_seeds

# Show selected tasks and seeds
selected_tasks = st.multiselect('Tasks selected', c.union_of_tasks, default=tasks)
selected_seeds = st.multiselect('Seeds selected', c.union_of_completed_seeds, default=seeds)

# Visualize
st.header('Results')
task = st.selectbox('Select task', ['all'] + selected_tasks)
task = selected_tasks if task == 'all' else [task]
show_std = st.checkbox('Show standard deviation')

st.subheader('Train')
c.visualize(data='train', tasks=task, seeds=selected_seeds, include_incomplete_files=include_incomplete_files, show_std=show_std)
st.pyplot()

st.subheader('Test')
c.visualize(data='test', tasks=task, seeds=selected_seeds, include_incomplete_files=include_incomplete_files, show_std=show_std)
st.pyplot()

st.subheader('Ranked train')
c.rank(data='train', tasks=task, seeds=selected_seeds, include_incomplete_files=include_incomplete_files, show_std=show_std)
st.pyplot()

st.subheader('Ranked test')
c.rank(data='train', tasks=task, seeds=selected_seeds, include_incomplete_files=include_incomplete_files, show_std=show_std)
st.pyplot()