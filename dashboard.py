import os

import streamlit as st

from benchmark.config import TASKS, SEEDS
from live import Collection

c = Collection()
c = c.add_files('baseline/')
# c = c.add_files('output/results-plus/')

# num_iterations = len(os.listdir('output/results-plus/'))
num_iterations = 3
num_seeds = len(SEEDS)

# Make a selection
st.sidebar.title('Selection')
st.sidebar.markdown(
    'Choose a subset to visualize in order to make the comparison between an existing and a running experiment fair.')
selections = {
    'complete': 'Use only completed iterations',
    'incomplete': 'Use completed tasks in incomplete iterations'
}
selection = st.sidebar.radio(
    'Subset for comparison',
    (selections['complete'], selections['incomplete'])
)

# Set the selection
tasks = c.all_tasks if selections['complete'] == selection else c.common_tasks
include_incomplete_files = selections['incomplete'] == selection

st.sidebar.title('Tasks')
st.sidebar.markdown("Choose the task you want to visualize.")
task = st.sidebar.selectbox(
    label='Task(s) to visualize',
    options=['all'] + tasks
)

if task == 'all':
    task = tasks
else:
    task = [task]

# Show the progress
st.sidebar.title(f"Progress")
st.sidebar.markdown(f"""
    - Current iteration: {num_iterations}/{num_seeds}
    - Tasks completed: {len(c.common_tasks)}/{len(TASKS)}
""")
st.sidebar.progress(len(c.common_tasks) / len(TASKS))

st.subheader('Training results')
c.visualize(data='train', tasks=task, include_incomplete_files=include_incomplete_files)
st.pyplot()
st.subheader('Validation results')
c.visualize(data='test', tasks=task, include_incomplete_files=include_incomplete_files)
st.pyplot()
