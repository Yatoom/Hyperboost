import numpy as np

from smac.optimizer.acquisition import AbstractAcquisitionFunction


class ScorePlusDistance(AbstractAcquisitionFunction):

    def __init__(self, model):
        super().__init__(model)

    def _compute(self, X: np.ndarray):
        score, distance = self.model.predict(X)
        return score + distance

        # Great when distance is not normalized by distance statistics (but still by the maximum possible distance)
        # return score + distance * 12
