import numpy as np
from smac.epm.util_funcs import get_types
from smac.facade.smac_ac_facade import SMAC4AC
from smac.initial_design.sobol_design import SobolDesign
from smac.optimizer.acquisition import LogEI
from smac.optimizer.ei_optimization import InterleavedLocalAndRandomSearch
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogScaledCost

from hyperboost.direct import Direct
from hyperboost.faster_ei_optimization import FasterInterleavedLocalAndRandomSearch
from hyperboost.lightepm import LightEPM
from hyperboost.lightepm_with_std import LightEPMWithStd
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from smac.utils.constants import MAXINT

class Hyperboost(SMAC4AC):
    def __init__(self, rng: np.random.RandomState = None, pca_components=None, **kwargs):
        scenario = kwargs['scenario']
        types, bounds = get_types(scenario.cs, scenario.feature_array)

        kwargs['initial_design'] = kwargs.get('initial_design', SobolDesign)
        kwargs['runhistory2epm'] = kwargs.get('runhistory2epm', RunHistory2EPM4LogScaledCost)

        init_kwargs = kwargs.get('initial_design_kwargs', dict())
        init_kwargs['n_configs_x_params'] = init_kwargs.get('n_configs_x_params', 10)
        init_kwargs['max_config_fracs'] = init_kwargs.get('max_config_fracs', 0.25)
        kwargs['initial_design_kwargs'] = init_kwargs

        # only 1 configuration per SMBO iteration
        intensifier_kwargs = kwargs.get('intensifier_kwargs', dict())
        intensifier_kwargs['min_chall'] = 1
        kwargs['intensifier_kwargs'] = intensifier_kwargs
        scenario.intensification_percentage = 1e-10

        model_class = LightEPM
        kwargs['model'] = model_class

        # == static RF settings
        model_kwargs = {
            "types": types,
            "bounds": bounds,
            "instance_features": scenario.feature_array,
            "seed": rng.randint(MAXINT),
            "pca_components_": pca_components,
            "configspace": scenario.cs
        }
        kwargs['model_kwargs'] = model_kwargs

        # == Acquisition function
        kwargs['acquisition_function'] = kwargs.get('acquisition_function', LogEI)

        kwargs['runhistory2epm'] = kwargs.get('runhistory2epm', RunHistory2EPM4LogScaledCost)

        # assumes random chooser for random configs
        random_config_chooser_kwargs = kwargs.get('random_configuration_chooser_kwargs', dict())
        random_config_chooser_kwargs['prob'] = random_config_chooser_kwargs.get('prob', 0.2)
        kwargs['random_configuration_chooser_kwargs'] = random_config_chooser_kwargs

        # better improve acquisition function optimization
        # 1. increase number of sls iterations
        acquisition_function_optimizer_kwargs = kwargs.get('acquisition_function_optimizer_kwargs', dict())
        acquisition_function_optimizer_kwargs['n_sls_iterations'] = 10
        kwargs['acquisition_function_optimizer_kwargs'] = acquisition_function_optimizer_kwargs

        super().__init__(**kwargs)
        self.logger.info(self.__class__)

        # better improve acquisition function optimization
        # 2. more randomly sampled configurations
        self.solver.scenario.acq_opt_challengers = 10000

        # activate predict incumbent
        self.solver.predict_incumbent = True


# class OldHyperboost(SMAC4AC):
#     def __init__(self, scenario: Scenario, rng: np.random.RandomState = None, method="default", pca_components=None,
#                  **kwargs):
#         types, bounds = get_types(scenario.cs, scenario.feature_array)
#
#         if method == "skopt":
#             model = LightEPMWithStd(types=types, bounds=bounds, instance_features=scenario.feature_array,
#                                     seed=rng.randint(MAXINT), pca_components=scenario.PCA_DIM)
#             super().__init__(scenario=scenario, rng=rng, model=model,
#                              acquisition_function_optimizer=InterleavedLocalAndRandomSearch, **kwargs)
#
#         elif method == "default":
#             acquisition_function = Direct
#             model = LightEPM
#             super().__init__(scenario=scenario, rng=rng,
#                              acquisition_function=acquisition_function,
#                              acquisition_function_optimizer=InterleavedLocalAndRandomSearch, **kwargs)
#             self.model = LightEPM(types=types, bounds=bounds, instance_features=scenario.feature_array,
#                              seed=rng.randint(MAXINT), pca_components_=pca_components, configspace=scenario.cs)
