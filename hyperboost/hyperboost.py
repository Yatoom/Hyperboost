import numpy as np
from smac.epm.util_funcs import get_types

from hyperboost.direct import Direct
from hyperboost.lightepm import LightEPM
from hyperboost.lightepm_with_std import LightEPMWithStd
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from smac.utils.constants import MAXINT


class Hyperboost(SMAC4HPO):
    def __init__(self, scenario: Scenario, rng: np.random.RandomState = None, method="default", pca_components=None,
                 **kwargs):
        types, bounds = get_types(scenario.cs, scenario.feature_array)

        if method == "skopt":
            model = LightEPMWithStd(types=types, bounds=bounds, instance_features=scenario.feature_array,
                                    seed=rng.randint(MAXINT), pca_components=scenario.PCA_DIM)
            super().__init__(scenario=scenario, rng=rng, model=model, **kwargs)

        elif method == "default":
            model = LightEPM(types=types, bounds=bounds, instance_features=scenario.feature_array,
                             seed=rng.randint(MAXINT), pca_components_=pca_components, configspace=scenario.cs)
            acquisition_function = Direct
            super().__init__(scenario=scenario, rng=rng, model=model,
                             acquisition_function=acquisition_function, **kwargs)
