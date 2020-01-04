import os

from sklearn.model_selection import KFold, ShuffleSplit
from benchmark.param_spaces import RandomForestSpace, GradientBoostingSpace, DecisionTreeSpace

"""
Seeds to make the HPO algorithm reproducible
"""
SEEDS = [2268061101, 2519249986, 338403738]

"""
Number of iterations in a HPO algorithm.
The maximum number of times we test a configuration. 
"""
NUM_ITER = 250

"""
Task IDs of tasks to load from OpenML. 
Each task in OpenML is associated with one dataset. Normally these task includes information about how the result 
should be tested, i.e. how the data should be split and which metric to use. We don't follow this task in our 
benchmarks, but simply use the dataset. Each task can be viewed at openml.org/t/<task_id>
"""
TASKS = [125920, 49, 146819, 29, 15, 3913, 3, 10101, 9971, 146818, 3917, 37, 3918, 14954, 9946, 146820, 3021, 31, 10093,
         3902, 3903, 9952, 9957, 167141, 14952, 9978, 3904, 43, 219, 14965, 7592]

"""
"""
BASE_DIRECTORY = "../output/"

"""
Output folder for SMAC (and by extension, Hyperboost)
Specifies the output-directory for all emerging files from SMAC, such as logging and results.
"""
SMAC_OUTPUT_FOLDER = os.path.join(BASE_DIRECTORY, "smac_output/")

"""
Output text file for the benchmark, with the purpose of tracking the progress of the benchmark.
"""
BENCHMARK_OUTPUT_FILE = os.path.join(BASE_DIRECTORY, "output.txt")

"""
Directory for the benchmark results files
"""
RESULTS_DIRECTORY = os.path.join(BASE_DIRECTORY, "results-plus/")

"""
Prefix for the benchmark results files
"""
RESULTS_PREFIX = 'plus'

"""
Maximum number of algorithm calls per configuration. SMAC's original default: 2000.
"""
MAXR = 5

"""
The metric the HPO algorithm needs to optimize for.
Will be passed to Scikit-Learn's cross validation scoring parameter.
"""
METRIC = 'balanced_accuracy'

"""
Parameter spaces to optimize.
"""
ML_ALGORITHMS = [
    RandomForestSpace(),
    # GradientBoostingSpace(),
    # DecisionTreeSpace()
]

"""
Splits for the outer loop.
The HPO algorithm will be executed using `1 - 1 / n_splits` part of the data. The resulting algorithm is also refitted 
on that part of the data, and then tested on the remaining data.
"""
OUTER_LOOP = KFold(n_splits=3, shuffle=True, random_state=42)

"""
Splits for the inner loop.
These are the splits used when testing out a configuration inside the HPO algorithm's loop. 
"""
INNER_LOOP = ShuffleSplit(n_splits=3, random_state=0, test_size=0.10, train_size=None)
