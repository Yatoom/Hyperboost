import time
import traceback
import typing

import numpy as np
from lightgbm import LGBMRegressor
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

from smac.epm.base_epm import AbstractEPM


class LightEPM(AbstractEPM):
    def __init__(self, types: np.ndarray, bounds: typing.List[typing.Tuple[float, float]],
                 instance_features: np.ndarray = None, pca_components=None, pca_components_: float = None, seed=None, configspace=None):

        super().__init__(types=types, bounds=bounds, instance_features=instance_features, pca_components=pca_components,
                         configspace=configspace, seed=seed)
        self.light = LGBMRegressor(verbose=-1, min_child_samples=1, objective="quantile", num_leaves=8,
                                   alpha=0.03, min_data_in_bin=1, n_jobs=1, n_estimators=100, random_state=seed)

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
        self.max_distance = sum(np.maximum(i, 1) for i in types) ** 2

        self.pca_components_ = pca_components_
        if pca_components_ is not None and pca_components_ > 0:
            self.pca_ = PCA(n_components=pca_components_)
        else:
            self.pca_ = None
        print("##Init")

    def _train(self, X, y):
        t0 = time.time()
        # print("##Train")
        # traceback.print_stack()
        X_ = X
        y_ = y

        self.X = X_
        self.y = y_
        self.X_transformed = self.transform(X)
        self.inc = np.max(y_)
        n_samples = X_.shape[0]

        if n_samples >= 2:
            self.light.fit(X_, y_.flatten())

            if self.pca_ is not None and self.X_transformed.shape[1] > self.pca_components_:
                self.X_transformed = self.pca_.fit_transform(self.X_transformed)

        self.kdtree = cKDTree(self.X_transformed)
        t1 = time.time()
        print((t1 - t0) * 1000)


    def _predict(self, X):

        # Zeros returned in case model was not fitted
        loss = np.zeros(X.shape[0])

        # Variance only returned for compatibility
        closeness = np.zeros(X.shape[0])

        # Model not fitted
        if self.light._n_features is None:
            return loss, closeness
        else:
            loss = self.light.predict(X)
            X_transformed = self.transform(X)

            if self.pca_ is not None and X_transformed.shape[1] > self.pca_components_ and \
                    self.X_transformed.shape[0] >= 2:
                X_transformed = self.pca_.transform(X_transformed)

            dist, ind = self.kdtree.query(X_transformed, k=1, p=2)

            scale = np.std(self.y)
            # print("var_y:", np.var(self.y), "var_x:", np.var(self.X))
            unscaled_dist = dist.reshape(-1) / self.max_distance
            # loss[unscaled_dist == 0] = 1
            dist = unscaled_dist * scale
            closeness = 1 - dist

        # print("##predict")
        return loss, closeness

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
