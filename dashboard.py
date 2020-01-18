from visualization import Collection, variables
import os
from benchmark import config
import streamlit as st


c = Collection()
c = c.add_files('b/')
output_dir = os.path.join('visualization', config.RESULTS_DIRECTORY)
if os.path.exists(output_dir):
    c = c.add_files(output_dir)

# SIDEBAR
st.sidebar.subheader('Filters')
target_name = st.sidebar.selectbox('Target algorithm', c.targets)

subset_type = st.sidebar.radio(
    'Subset to visualize',
    (variables.REMOVE_INCOMPLETE, variables.TASK_INTERSECTION)
)

tasks = c.common_tasks if subset_type == variables.TASK_INTERSECTION else c.tasks
tasks = sorted(list(tasks))

st.sidebar.subheader('Visualization')
show_std = st.sidebar.checkbox('Show standard deviation')
show_input = st.sidebar.checkbox('Show input for visualization')
task = st.sidebar.selectbox('Select task(s) to display', ['all'] + tasks)

# Apply filters
tasks = tasks if task == 'all' else [task]
c = c.filter(tasks, target_name)

# Input
if show_input:
    st.table(c.overview)
    st.markdown(f"""
        **Target algorithm: ** {target_name}    
        **{len(tasks)} tasks selected: ** {', '.join([f'[{i}](http://openml.org/t/{i})' for i in tasks])}
    
        ---
    """)

c.visualize(data='train', show_std=show_std)
c.visualize(data='test', show_std=show_std)
c.visualize(data='train', ranked=True, show_std=show_std)
c.visualize(data='test', ranked=True, show_std=show_std)