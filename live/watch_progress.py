import copy
import json
from collections import defaultdict
import os
import numpy as np
from live import Group, File, Collection


c = Collection()
c = c.add_files('../testing/')
c = c.add_files('../testing2/')
print(c.common_tasks)
c.visualize(data='train', tasks=['3'])