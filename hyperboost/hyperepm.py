import typing
from math import sqrt
from pprint import pprint

import numpy as np
from lightgbm import LGBMRegressor
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from smac.epm.base_epm import AbstractEPM


class HyperEPM(AbstractEPM):
    def __init__(self, types: np.ndarray, bounds: typing.List[typing.Tuple[float, float]],
                 instance_features: np.ndarray = None, pca_components_: float = None, seed=None, configspace=None):

        print("HyperEPM!")
        super().__init__(types=types, bounds=bounds, instance_features=instance_features, pca_components=None,
                         configspace=configspace, seed=seed)
        self.light = LGBMRegressor(verbose=-1, min_child_samples=1, objective="quantile", num_leaves=8,
                                   alpha=0.10, min_data_in_bin=1, n_jobs=4, n_estimators=100, random_state=seed)

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

        # The maximum euclidean distance between two points in hyperparameter space
        lengths = np.array([np.maximum(i, 1) for i in types])
        self.max_distance = self.max_euclid_distance(lengths)

        self.pca_components_ = pca_components_

        if pca_components_ is not None and pca_components_ > 0:
            self.pca_ = PCA(n_components=pca_components_)
        else:
            self.pca_ = None

        self.distance_stats = None

    def _train(self, X, y):
        self.X = X
        self.y = y

        self.y_scaler = MinMaxScaler(feature_range=(-1,0)).fit(y)

        # One-hot-encode the categorical variables
        self.X_transformed = self.transform(X)

        self.inc = np.min(y)  # minimize loss
        n_samples = X.shape[0]

        if n_samples >= 2:
            self.light.fit(X, y.flatten())

            if self.pca_ is not None and self.X_transformed.shape[1] > self.pca_components_:
                self.X_transformed = self.pca_.fit_transform(self.X_transformed)

                # Update maximum distance because the axes in PCA might cause slightly different maximum distance
                # new_max_distance = self.max_euclid_distance(self.lengths(self.X_transformed))
                # self.max_distance = new_max_distance

        self.kdtree = cKDTree(self.X_transformed)
        self.distance_stats = self._estimate_distance_statistics()
        pprint(self.distance_stats)

    # def _avg_distance(self):
    #     points = np.random.random((10000, self.X_transformed.shape[1]))
    #     distances, indices = self.kdtree.query(points, k=1)
    #     return np.mean(distances)

    def _estimate_distance_statistics(self):

        sample = self.configspace.sample_configuration(10000)
        sample = [i._vector for i in sample]

        # One-hot-encode sample
        sample_transformed = self.transform(sample)

        sample = self.pca_.transform(sample_transformed)

        distances, indices = self.kdtree.query(sample, k=1)
        return {
            'max_possible': self.max_distance,
            'mean': np.mean(distances),
            'max': np.max(distances),
            'min': np.min(distances),  # Should be close to zero
            'median': np.median(distances),
            'std': np.std(distances)
        }

    def _predict(self, X):

        # Return zero's if the model is not fitted
        if self.light._n_features is None:
            return np.zeros(X.shape[0]), np.zeros(X.shape[0])

        # Predict the loss of each sample in X
        loss = self.light.predict(X)

        # One-hot-encode X
        X_transformed = self.transform(X)

        # Perform dimensionality reduction on X if its transformed
        # (i.e. one-hot-encoded) form contains more dimensions
        # than the maximum we specified.
        if (
            self.pca_ is not None and
            X_transformed.shape[1] > self.pca_components_ and
            self.X_transformed.shape[0] >= 2
        ):
            X_transformed = self.pca_.transform(X_transformed)

        # Calculate the distance to the closest known sample
        distances, indices = self.kdtree.query(X_transformed, k=1, p=2)

        # Transform distance to scale 0-1.
        normalized_distance = distances.reshape(-1) / self.distance_stats['max']  # Replace by avg_distance here?

        # Transform loss to scale (-1, 0).
        normalized_loss = self.y_scaler.transform(np.atleast_2d(loss))[0]

        # print("distance", normalized_distance.min(), normalized_distance.max())
        # print("loss", normalized_loss.min(), normalized_loss.max())

        # Scale distance
        # scale = np.std(self.y)
        # scaled_distance = normalized_distance * scale

        # Return score and distance
        return -normalized_loss, normalized_distance

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

    @staticmethod
    def lengths(X: np.array) -> np.array:
        return X.max(axis=0) - X.min(axis=0)

    @staticmethod
    def max_euclid_distance(lengths: np.array) -> np.array:
        return np.sqrt(np.sum(lengths**2))
