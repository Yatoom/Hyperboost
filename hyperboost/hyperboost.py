import numpy as np

from hyperboost.direct import Direct, DirectNoDistance
from hyperboost.lightdropepm import LightDropEPM
from hyperboost.lightepm import LightEPM
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.utils.constants import MAXINT
from smac.utils.util_funcs import get_types


class Hyperboost(SMAC):
    def __init__(self, scenario: Scenario, rng: np.random.RandomState = None, method="drop", **kwargs):
        types, bounds = get_types(scenario.cs, scenario.feature_array)

        if method == "drop":
            model = LightDropEPM(types=types, bounds=bounds, instance_features=scenario.feature_array,
                                 seed=rng.randint(MAXINT), pca_components=scenario.PCA_DIM)
            super().__init__(scenario=scenario, rng=rng, model=model, use_pynisher=False, **kwargs)
        elif method == "QRD":
            model = LightEPM(types=types, bounds=bounds, instance_features=scenario.feature_array,
                             seed=rng.randint(MAXINT), pca_components=scenario.PCA_DIM)
            acquisition_function = Direct(model=model)
            super().__init__(scenario=scenario, rng=rng, model=model, use_pynisher=False,
                             acquisition_function=acquisition_function, **kwargs)
        elif method == "QR":
            model = LightEPM(types=types, bounds=bounds, instance_features=scenario.feature_array,
                             seed=rng.randint(MAXINT), pca_components=scenario.PCA_DIM)
            acquisition_function = DirectNoDistance(model=model)
            super().__init__(scenario=scenario, rng=rng, model=model, use_pynisher=False,
                             acquisition_function=acquisition_function, **kwargs)
