from typing import List, Optional, Tuple

import numpy as np

from smac.configspace import ConfigurationSpace
from smac.epm.base_epm import BaseEPM
from catboost import CatBoostRegressor


class CatboostEPM(BaseEPM):
    """EPM which based on Catboost.

    Parameters
    ----------
    configspace : ConfigurationSpace
        Configuration space to tune for.
    types : List[int]
        Specifies the number of categorical values of an input dimension where
        the i-th entry corresponds to the i-th input dimension. Let's say we
        have 2 dimension where the first dimension consists of 3 different
        categorical choices and the second dimension is continuous than we
        have to pass [3, 0]. Note that we count starting from 0.
    bounds : List[Tuple[float, float]]
        bounds of input dimensions: (lower, uppper) for continuous dims; (n_cat, np.nan) for categorical dims
    seed : int
        The seed that is passed to the model library.
    instance_features : np.ndarray (I, K), optional
        Contains the K dimensional instance features
        of the I different instances
    pca_components : float
        Number of components to keep when using PCA to reduce
        dimensionality of instance features. Requires to
        set n_feats (> pca_dims).
    """

    def __init__(
            self,
            configspace: ConfigurationSpace,
            types: List[int],
            bounds: List[Tuple[float, float]],
            seed: int,
            instance_features: Optional[np.ndarray] = None,
            pca_components: Optional[int] = None,
    ) -> None:
        super().__init__(
            configspace=configspace,
            types=types,
            bounds=bounds,
            seed=seed,
            instance_features=instance_features,
            pca_components=pca_components,
        )
        self.rng = np.random.RandomState(self.seed)

        # Seems to work slightly better after 100 iterations.

        # V2
        # self.catboost = CatBoostRegressor(iterations=100, loss_function='RMSEWithUncertainty', posterior_sampling=True,
        #                                    verbose=False, random_seed=0, learning_rate=0.3)

        # V3
        # self.catboost = CatBoostRegressor(iterations=100, loss_function='RMSEWithUncertainty', posterior_sampling=False,
        #                                   verbose=False, random_seed=0, learning_rate=0.3)

        # V4
        self.catboost = CatBoostRegressor(iterations=100, loss_function="RMSEWithUncertainty", posterior_sampling=False,
                                          verbose=False, random_seed=0, learning_rate=0.5,
                                          )

        return None

    def _train(self, X: np.ndarray, Y: np.ndarray) -> "CatboostEPM":
        """Pseudo training on X and Y.

        Parameters
        ----------
        X : np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        Y : np.ndarray (N, 1)
            The corresponding target values.
        """

        if not isinstance(X, np.ndarray):
            raise NotImplementedError("X has to be of type np.ndarray")
        if not isinstance(Y, np.ndarray):
            raise NotImplementedError("Y has to be of type np.ndarray")

        self.catboost.fit(X, Y)
        # print(self.catboost.get_all_params())

        self.logger.debug("Fit model to data")
        return self

    def _predict(self, X: np.ndarray, cov_return_type: Optional[str] = "diagonal_cov") -> Tuple[np.ndarray, np.ndarray]:
        """Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples, n_features (config + instance features)]
        cov_return_type: Optional[str]
            Specifies what to return along with the mean. Refer ``predict()`` for more information.

        Returns
        -------
        means : np.ndarray of shape = [n_samples, n_objectives]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, n_objectives]
            Predictive variance
        """

        if cov_return_type != "diagonal_cov":
            raise ValueError("'cov_return_type' can only take 'diagonal_cov' for this model")

        if not isinstance(X, np.ndarray):
            raise NotImplementedError("X has to be of type np.ndarray")

        pred = self.catboost.predict(X)
        preds = self.catboost.virtual_ensembles_predict(X, prediction_type='TotalUncertainty',
                                                        virtual_ensembles_count=20)
        mean_preds = preds[:, 0]  # mean values predicted by a virtual ensemble
        knowledge = preds[:, 1]  # knowledge uncertainty predicted by a virtual ensemble
        data = preds[:, 2]  # average estimated data uncertainty

        return pred[:, 0], knowledge ** 0.3
        # Knowledge uncertainty at 0.3 seems fine.
        # Knowledge uncertainty is reduced too much at 0.4 --> Too exploitative (goes down fast, but then slows down)
        # Knowledge uncertainty at 0.2 seems to be slightly too explorative (takes too long to go down)

