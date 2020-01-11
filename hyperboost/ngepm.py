from abc import ABC

import typing
from smac.epm.base_epm import AbstractEPM
import numpy as np
from ngboost import NGBRegressor


class NGEPM(AbstractEPM, ABC):
    def __init__(self, types: np.ndarray, bounds: typing.List[typing.Tuple[float, float]],
                 instance_features: np.ndarray = None, seed=None, configspace=None):
        print("HyperEPM!")
        super().__init__(types=types, bounds=bounds, instance_features=instance_features, pca_components=None,
                         configspace=configspace, seed=seed)
        self.ngb = NGBRegressor()

    def _train(self, X: np.ndarray, Y: np.ndarray) -> 'AbstractEPM':
        self.ngb.fit(X, Y.flatten())
        print('trained')
        return self

    def _predict(self, X):
        y_dists = self.ngb.pred_dist(X)
        mean = y_dists.loc
        var = y_dists.var
        print('predicted')
        return mean, var
