import typing

import numpy as np
from lightgbm import LGBMRegressor

from smac.epm.base_epm import AbstractEPM


class LightEPMWithStd(AbstractEPM):
    def __init__(self, types: np.ndarray, bounds: typing.List[typing.Tuple[float, float]],
                 instance_features: np.ndarray = None, pca_components: float = None, seed=None):
        super().__init__(types=types, bounds=bounds, instance_features=instance_features, pca_components=pca_components)

        self.light_16 = LGBMRegressor(verbose=-1, min_child_samples=1, objective="quantile", num_leaves=8,
                                      alpha=0.16, min_data_in_bin=1, n_jobs=4, n_estimators=100, random_state=seed)
        self.light_84 = LGBMRegressor(verbose=-1, min_child_samples=1, objective="quantile", num_leaves=8,
                                      alpha=0.84, min_data_in_bin=1, n_jobs=4, n_estimators=100, random_state=seed)
        self.light_50 = LGBMRegressor(verbose=-1, min_child_samples=1, objective="quantile", num_leaves=8,
                                      alpha=0.50, min_data_in_bin=1, n_jobs=4, n_estimators=100, random_state=seed)

        # Observed values
        self.X = None
        self.y = None

        # The incumbent value
        self.inc = None

    def _train(self, X, y):
        self.X = X
        self.y = y.flatten()
        self.inc = np.max(y)

        n_samples = X.shape[0]
        if n_samples >= 2:
            self.light_16.fit(X, 1 - self.y)
            self.light_50.fit(X, 1 - self.y)
            self.light_84.fit(X, 1 - self.y)

        return self

    def _predict(self, X):

        # Zeros returned in case model was not fitted
        loss = np.zeros(X.shape[0])
        std = np.ones(X.shape[0])

        # Model fitted
        if self.light_16._n_features is not None:
            low = self.light_16.predict(X)
            high = self.light_84.predict(X)
            mean = self.light_50.predict(X)
            # std = (high - low) / 2.0
            std_reversed = (low - high) / 2.0
            return 1 - mean, std_reversed

        return loss, std
