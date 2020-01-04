from enum import Enum

import numpy as np
from smac.epm.util_funcs import get_types
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from smac.utils.constants import MAXINT

from hyperboost.acquistion_function import ScorePlusDistance
from hyperboost.hyperepm import HyperEPM
from hyperboost.skoptepm import SkoptEPM


class Method(Enum):
    # The default HyperBoost method. Builds 1 Gradient Boosting model to predict quantile 90 or 95. Combines this
    # prediction with distance from candidate to nearest observed, to prefer more sparse regions.
    HYPERBOOST = 1

    # Simulate Scikit-Optimize's way of using Gradient Boosting. Builds 3 Gradient Boosting models to predict quantiles
    # 16, 84 and 50. The difference between quantile 16 and 84 divided by two is the standard deviation.
    SCIKIT_OPTIMIZE = 2


class Hyperboost(SMAC4HPO):
    def __init__(self, scenario: Scenario, rng: np.random.RandomState = None, method=Method.HYPERBOOST,
                 pca_components=None, **kwargs):

        # Types and bounds required to initialize EPM
        types, bounds = get_types(scenario.cs, scenario.feature_array)

        if method == Method.SCIKIT_OPTIMIZE:

            # Initialize Scikit-optimize's empirical performance model
            model = SkoptEPM(types=types, bounds=bounds, instance_features=scenario.feature_array,
                             seed=rng.randint(MAXINT), pca_components=scenario.PCA_DIM)

            # Pass parameters to SMAC4HPO
            super().__init__(scenario=scenario, rng=rng, model=model, **kwargs)

            # Required for the newer version of SMAC.
            self.solver.model = model

        elif method == Method.HYPERBOOST:

            # Initialize HyperBoost's empirical performance model
            model = HyperEPM(types=types, bounds=bounds, instance_features=scenario.feature_array,
                             seed=rng.randint(MAXINT), pca_components_=pca_components, configspace=scenario.cs)

            # Pass parameters to SMAC4HPO
            super().__init__(scenario=scenario, rng=rng, model=model,
                             acquisition_function=ScorePlusDistance, **kwargs)

            # Required for the newer version of SMAC.
            self.solver.model = model
