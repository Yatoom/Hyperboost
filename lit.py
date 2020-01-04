import os

import streamlit as st

from benchmark.config import TASKS, SEEDS
from live import Collection

c = Collection()
c = c.add_files('testing/')
c = c.add_files('output/results-plus/')
num_iterations = len(os.listdir('output/results'))
num_seeds = len(SEEDS)
print(c.all_tasks)

st.sidebar.title('Tasks')
st.sidebar.markdown("Choose the task you want to visualize.")
task = st.sidebar.selectbox(
    label='',
    options=['in common', 'all'] + c.common_tasks
)

if task == 'all':
    task = c.all_tasks
elif task == 'in common':
    task = c.common_tasks
else:
    task = [task]

st.sidebar.title(f"Current iteration ({num_iterations}/{num_seeds})")
st.sidebar.markdown(f"Number of tasks completed in current iteration: {len(c.common_tasks)}/{len(TASKS)}")
st.sidebar.progress(len(c.common_tasks) / len(TASKS))

st.title('Training results')
# c.visualize_streamlit()
c.visualize(data='train', tasks=task)
st.pyplot()
st.title('Validation results')
# c.visualize_streamlit(data='train', tasks=task)
c.visualize(data='test', tasks=task)
st.pyplot()