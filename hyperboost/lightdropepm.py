import typing

import numpy as np
from lightgbm import LGBMRegressor

from smac.epm.base_epm import AbstractEPM


class LightDropEPM(AbstractEPM):
    def __init__(self, types: np.ndarray, bounds: typing.List[typing.Tuple[float, float]],
                 instance_features: np.ndarray = None, pca_components: float = None, seed=None, boosting_type="gbdt"):
        super().__init__(types=types, bounds=bounds, instance_features=instance_features, pca_components=pca_components)
        self.light = LGBMRegressor(verbose=-1, boosting_type="rf", min_child_samples=1, num_leaves=32,
                                   min_data_in_bin=1, n_jobs=4, n_estimators=100, random_state=seed,
                                   subsample_freq=1, subsample=0.66)

        # A KDTree to be constructed for measuring distance
        self.kdtree = None

        # Types and bounds
        self.bounds = bounds
        self.types = types

        # Observed values
        self.X = None
        self.y = None
        self.X_transformed = None

        # The incumbent value
        self.inc = None

        # Selection of hyperparameters that require one-hot-encoding
        self.selection = types != 0

        # Flag that checks if there are any nominal parameters
        self.contains_nominal = any(self.selection)

        # The number of possible categories per nominal parameter
        self.categories = types[self.selection]

        # The maximum L1 distance between two points in hyperparameter space
        self.max_distance = sum(np.maximum(i, 1) for i in types)

    def _train(self, X, y):
        X_ = X
        y_ = y

        self.X = X_
        self.y = y_
        self.inc = np.max(y_)
        n_samples = X_.shape[0]
        if n_samples >= 2:
            self.light.fit(X_, y_.flatten())

    def _predict(self, X):

        # Zeros returned in case model was not fitted
        loss = np.zeros(X.shape[0])

        # Variance only returned for compatibility
        sigma = np.zeros(X.shape[0])

        # Model not fitted
        if self.light._n_features is None:
            return loss, sigma
        else:
            loss = self.light.predict(X)
            # individual = self._get_individual_predictions(self.light, X)
            sigma = self._get_uncertainty(self.light, X)

        return loss, sigma

    @staticmethod
    def _get_uncertainty(lgbm, data):
        num_trees = lgbm._Booster.num_trees()

        all_predictions = []
        for _ in range(5):
            shuffled = lgbm._Booster.shuffle_models(start_iteration=0)
            predictions = shuffled.predict(data, int(num_trees/2))
            all_predictions.append(predictions)

        return np.var(all_predictions, axis=0)

    @staticmethod
    def _get_individual_predictions(lgbm, data):
        predictions = lgbm.predict(data, pred_leaf=True)
        return [[lgbm._Booster.get_leaf_output(i, leaf) for i, leaf in enumerate(i_preds)] for i_preds in predictions]

    @staticmethod
    def one_hot_vector(length, indicator):
        result = np.zeros(length)
        result[indicator] = 1
        return result
