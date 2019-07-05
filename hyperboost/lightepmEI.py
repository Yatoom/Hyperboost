import typing

import numpy as np
from lightgbm import LGBMRegressor
from scipy.spatial import cKDTree
from scipy.stats import norm
from sklearn.exceptions import NotFittedError

from smac.epm.base_epm import AbstractEPM


class LightEPMEI(AbstractEPM):
    def __init__(self, types: np.ndarray, bounds: typing.List[typing.Tuple[float, float]],
                 instance_features: np.ndarray = None, pca_components: float = None, seed=None):
        super().__init__(types=types, bounds=bounds, instance_features=instance_features, pca_components=pca_components)
        self.light = LGBMRegressor(verbose=-1, min_child_samples=1, objective="quantile", num_leaves=8,
                                   alpha=0.90, min_data_in_bin=1, n_jobs=4, n_estimators=100, random_state=seed)

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
        self.pos_y = 1 - y_.flatten()
        self.X_transformed = self.transform(X)
        self.kdtree = cKDTree(self.X_transformed)
        self.inc = np.max(y_)
        n_samples = X_.shape[0]
        if n_samples >= 2:
            self.light.fit(X_, self.pos_y)

    def _predict(self, X):

        # Predict with estimator and query KD-tree
        try:
            q_val = self.light.predict(X)
            dist, ind = self.kdtree.query(self.transform(X), k=1, p=1)
        except NotFittedError:
            return np.zeros(len(X)), np.zeros(len(X))

        # Get the scores of the nearest neighbor
        neighbor_score = np.array(self.pos_y)[ind]
        mean = neighbor_score

        # Calculate statistics
        y_max = np.max(self.pos_y)
        y_std = np.std(self.pos_y)

        # Normalize distance
        dist = dist.reshape(-1) / self.max_distance * y_std

        # Determine uncertainty values
        aleatory = np.maximum(0, q_val - neighbor_score)
        epistemic = dist
        # var = aleatory * epistemic
        # std = aleatory * epistemic + epistemic**2

        var = (np.sqrt(aleatory * epistemic) + epistemic) ** 2
        # OR
        # var = aleatory * epistemic + epistemic ** 2

        # aleatory = q_val - neighbor_score
        # epistemic = dist * neighbor_score / y_max
        # std = (aleatory + epistemic) / 2

        return 1 - mean, var

    def transform(self, X):
        if not self.contains_nominal:
            return X

        result = []
        for i in X:
            # Split
            nominal = i[self.selection].astype(int)
            numerical = i[~self.selection]

            # Concatenate one-hot encoded together with numerical
            r = np.concatenate(
                [self.one_hot_vector(self.categories[index], indicator) for index, indicator in enumerate(nominal)])
            r = np.concatenate([numerical, r])

            result.append(r)

        return np.array(result)

    @staticmethod
    def one_hot_vector(length, indicator):
        result = np.zeros(length)
        result[indicator] = 1
        return result
