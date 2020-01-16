import os

import streamlit as st
import pandas as pd

from benchmark.config import TASKS, SEEDS
from visualization import Collection, variables

c = Collection()
c = c.add_files('b/')
# c = c.add_files('output/results4/')

num_iterations = len(os.listdir('output/results3/'))
num_seeds = len(SEEDS)

# Show overview of the data we have available
st.table(c.overview)

# Make a subset of the data
st.sidebar.subheader('Target algorithm')
target_model = st.sidebar.selectbox('Target algorithm', c.target_models)

st.sidebar.subheader('Selection')
subset_type = st.sidebar.radio(
    'Choose a subset to visualize in order to make the comparison between an existing and a running experiment fair.',
    (variables.REMOVE_INCOMPLETE, variables.TASK_INTERSECTION))
subset_iterations = st.sidebar.radio(
    'From the subset above, for each experiment:',
    (variables.ALL_SEEDS, variables.SEED_INTERSECTION)
)

# Set the selection
tasks = c.intersection_of_tasks if subset_type == variables.TASK_INTERSECTION else c.union_of_tasks
include_incomplete_files = subset_type != variables.REMOVE_INCOMPLETE
seeds = c.union_of_completed_seeds if subset_iterations == variables.ALL_SEEDS else c.intersection_of_completed_seeds

# Show selected tasks and seeds
# selected_tasks = st.multiselect('Tasks selected', c.union_of_tasks, default=sorted(tasks))
# selected_seeds = st.multiselect('Seeds selected', c.union_of_completed_seeds, default=sorted(seeds))

# Sort for consistent display
tasks = sorted(tasks)
seeds = sorted(seeds)

# Visualize
st.sidebar.subheader('Display')
show_std = st.sidebar.checkbox('Show standard deviation')

st.sidebar.subheader('Task(s)')
task = st.sidebar.selectbox('Select task(s) to display', ['all'] + tasks)
task = tasks if task == 'all' else [task]

st.markdown(f"""
    **Target algorithm: ** {target_model}    
    **{len(seeds)} seeds selected: ** {', '.join([str(s) for s in seeds])}    
    **{len(task)} tasks selected: ** {', '.join([f'[{i}](http://openml.org/t/{i})' for i in task])}

    ---
""")

# st.markdown(f'**{len(seeds)} seeds selected:** ' + ', '.join([str(s) for s in seeds]))
# st.markdown(f'**{len(task)} task(s) selected:**')
# st.markdown(', '.join([f'[{i}](http://openml.org/t/{i})' for i in task]))


st.subheader('Training loss')
c.visualize(data='train', tasks=task, seeds=seeds, include_incomplete_files=include_incomplete_files,
            show_std=show_std, target_model=target_model)
st.pyplot()

st.subheader('Testing loss')
c.visualize(data='test', tasks=task, seeds=seeds, include_incomplete_files=include_incomplete_files,
            show_std=show_std, target_model=target_model)
st.pyplot()

st.subheader('Training ranks')
c.visualize(data='train', tasks=task, seeds=seeds, include_incomplete_files=include_incomplete_files,
            show_std=show_std, target_model=target_model, ranked=True)
st.pyplot()

st.subheader('Testing ranks')
c.visualize(data='test', tasks=task, seeds=seeds, include_incomplete_files=include_incomplete_files,
            show_std=show_std, target_model=target_model, ranked=True)
st.pyplot()
