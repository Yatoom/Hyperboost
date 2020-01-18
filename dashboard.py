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
show_input = st.sidebar.checkbox('Show overview of experiments')
task = st.sidebar.selectbox('Select task(s) to display', ['all'] + tasks)

# Apply filters
tasks = tasks if task == 'all' else [task]
c = c.filter(tasks, target_name)

# Input
if show_input:
    st.subheader("Overview of experiments")
    st.markdown(f"""
        The visualization includes experiments for target function/algorithm `{target_name}` using  `{len(tasks)}` 
        tasks: {', '.join([f'[{i}](http://openml.org/t/{i})' for i in tasks])}. The table below shows the metadata 
        for each file included in the benchmark results. This metadata is derived from the filenames, which follow 
        the form: `<Directory>/<Experiment>-<Target>-<Iter. seed>.json`.
     """)
    st.table(c.overview)


st.subheader("Average loss")
st.markdown("""
    The graphs below show train and test losses. The standard deviation of an HPO-experiment is calculated on 
    task-level over each benchmark iteration. These standard deviations are then averaged over all tasks.
""")
c.visualize(data='train', show_std=show_std)
c.visualize(data='test', show_std=show_std)

st.subheader("Average ranking")
st.markdown("""
    The graphs below show train and test ranks. The ranks are calculated per benchmark iteration, where missing 
    iterations are temporarily filled in with last available iteration. The ranks are then averaged over the benchmark 
    iterations and tasks. The standard deviation of each HPO-experiment is calculated globally over each task.
""")
c.visualize(data='train', ranked=True, show_std=show_std, global_std=True)
c.visualize(data='test', ranked=True, show_std=show_std, global_std=True)