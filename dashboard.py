from visualization import Collection, variables
import os
from benchmark import config
import streamlit as st


c = Collection()
c = c.add_files('baseline/')
c = c.add_files('output/results/')
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
show_input = st.sidebar.checkbox('Show overview of experiments', value=True)
task = st.sidebar.selectbox('Select task(s) to display', ['all'] + tasks)

# Apply filters
tasks = tasks if task == 'all' else [task]
c = c.filter(tasks, target_name)

# for g in c.groups:
#     st.sidebar.checkbox(f"{g.prefix}")

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

baseline = "baseline/benchmark-smac"
st.subheader("% wins against baseline")
st.table(c.result_table(baseline))
st.subheader("% wins over time against baseline")
st.plotly_chart(c.visualize_wins(baseline))


st.subheader("Average loss")
st.markdown("""
    The graphs below show train and test losses. The standard deviation of an HPO-experiment is calculated on 
    task-level over each benchmark iteration. These standard deviations are then averaged over all tasks.
""")
train_graph, table = c.visualize(data='train', show_std=show_std)
test_graph, _ = c.visualize(data='test',  show_std=show_std)
st.plotly_chart(train_graph)
st.plotly_chart(test_graph)

st.subheader("Average ranking")
st.markdown("""
    The graphs below show train and test ranks. The ranks are calculated per benchmark iteration, where missing 
    iterations are temporarily filled in with last available iteration. The ranks are then averaged over the benchmark 
    iterations and tasks. The standard deviation of each HPO-experiment is calculated globally over each task.
""")
train_graph, _ = c.visualize(data='train', ranked=True, show_std=show_std, global_std=True)
test_graph, _ = c.visualize(data='test', ranked=True, show_std=show_std, global_std=True)
st.plotly_chart(train_graph)
st.plotly_chart(test_graph)

st.subheader("Average run times")
st.markdown("""
    The table below shows the average run time for the HPO-experiment per task. The `avg. target eval. time` is the 
    average time that is spend on running the target algorithm/function to evaluate a hyperparameter configuration 
    sample. The standard deviation the run times are calculated on task-level over each benchmark iteration. 
    These standard deviations are then averaged over all tasks.
""")
st.table(table)