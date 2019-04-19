import numpy as np

from hyperboost.direct import Direct
from hyperboost.lightepm import LightEPM
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.utils.constants import MAXINT
from smac.utils.util_funcs import get_types


class Hyperboost(SMAC):
    def __init__(self, scenario: Scenario, rng: np.random.RandomState = None, **kwargs):
        types, bounds = get_types(scenario.cs, scenario.feature_array)
        model = LightEPM(types=types, bounds=bounds, instance_features=scenario.feature_array,
                         seed=rng.randint(MAXINT), pca_components=scenario.PCA_DIM)
        acquisition_function = Direct(model=model)
        super().__init__(scenario=scenario, rng=rng, model=model, acquisition_function=acquisition_function,
                         use_pynisher=False, **kwargs)
