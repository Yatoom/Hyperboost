import numpy as np

from smac.optimizer.acquisition import AbstractAcquisitionFunction


class ScorePlusDistance(AbstractAcquisitionFunction):

    def __init__(self, model):
        super().__init__(model)

    def _compute(self, X: np.ndarray):
        loss, closeness = self.model.predict(X)
        return 1 - (loss + closeness) / 2
