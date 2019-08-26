import numpy as np

from hyperboost.direct import Direct, DirectNoDistance
from hyperboost.lightcombo import LightCombo
from hyperboost.lightdropepm import LightDropEPM
from hyperboost.lightepm import LightEPM
from hyperboost.lightepm2 import LightEPM2
from hyperboost.lightepmEI import LightEPMEI
from hyperboost.lightepm_with_std import LightEPMWithStd
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.utils.constants import MAXINT
from smac.utils.util_funcs import get_types


class Hyperboost(SMAC):
    def __init__(self, scenario: Scenario, rng: np.random.RandomState = None, method="default", **kwargs):
        types, bounds = get_types(scenario.cs, scenario.feature_array)

        if method == "skopt":
            model = LightEPMWithStd(types=types, bounds=bounds, instance_features=scenario.feature_array,
                                        seed=rng.randint(MAXINT), pca_components=scenario.PCA_DIM)
            super().__init__(scenario=scenario, rng=rng, model=model, use_pynisher=False, **kwargs)

        elif method == "default":
            model = LightEPM(types=types, bounds=bounds, instance_features=scenario.feature_array,
                             seed=rng.randint(MAXINT), pca_components=scenario.PCA_DIM)
            acquisition_function = Direct(model=model)
            super().__init__(scenario=scenario, rng=rng, model=model, use_pynisher=False,
                             acquisition_function=acquisition_function, **kwargs)

        # if method == "skopt":
        #     model = LightEPMWithStd(types=types, bounds=bounds, instance_features=scenario.feature_array,
        #                             seed=rng.randint(MAXINT), pca_components=scenario.PCA_DIM)
        #     super().__init__(scenario=scenario, rng=rng, model=model, use_pynisher=False, **kwargs)
        #
        # if method == "drop":
        #     model = LightDropEPM(types=types, bounds=bounds, instance_features=scenario.feature_array,
        #                          seed=rng.randint(MAXINT), pca_components=scenario.PCA_DIM, boosting_type="gbdt")
        #     super().__init__(scenario=scenario, rng=rng, model=model, use_pynisher=False, **kwargs)
        # elif method == "drop-dart":
        #     model = LightDropEPM(types=types, bounds=bounds, instance_features=scenario.feature_array,
        #                          seed=rng.randint(MAXINT), pca_components=scenario.PCA_DIM, boosting_type="dart")
        #     super().__init__(scenario=scenario, rng=rng, model=model, use_pynisher=False, **kwargs)
        # elif method == "combo":
        #     model = LightCombo(types=types, bounds=bounds, instance_features=scenario.feature_array,
        #                        seed=rng.randint(MAXINT), pca_components=scenario.PCA_DIM)
        #     super().__init__(scenario=scenario, rng=rng, model=model, use_pynisher=False, **kwargs)
        # elif method == "QRD":
        #     model = LightEPM(types=types, bounds=bounds, instance_features=scenario.feature_array,
        #                      seed=rng.randint(MAXINT), pca_components=scenario.PCA_DIM, scaling=scaling)
        #     acquisition_function = Direct(model=model)
        #     super().__init__(scenario=scenario, rng=rng, model=model, use_pynisher=False,
        #                      acquisition_function=acquisition_function, **kwargs)
        # elif method == "QR":
        #     model = LightEPM(types=types, bounds=bounds, instance_features=scenario.feature_array,
        #                      seed=rng.randint(MAXINT), pca_components=scenario.PCA_DIM)
        #     acquisition_function = DirectNoDistance(model=model)
        #     super().__init__(scenario=scenario, rng=rng, model=model, use_pynisher=False,
        #                      acquisition_function=acquisition_function, **kwargs)
        # elif method == "EI":
        #     model = LightEPMEI(types=types, bounds=bounds, instance_features=scenario.feature_array,
        #                        seed=rng.randint(MAXINT), pca_components=scenario.PCA_DIM)
        #     super().__init__(scenario=scenario, rng=rng, model=model, use_pynisher=False, **kwargs)
        # elif method == "2":
        #     model = LightEPM2(types=types, bounds=bounds, instance_features=scenario.feature_array,
        #                      seed=rng.randint(MAXINT), pca_components=scenario.PCA_DIM, scaling=scaling)
        #     acquisition_function = Direct(model=model)
        #     super().__init__(scenario=scenario, rng=rng, model=model, use_pynisher=False,
        #                      acquisition_function=acquisition_function, **kwargs)
