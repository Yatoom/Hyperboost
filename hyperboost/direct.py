import numpy as np

from smac.epm.base_epm import AbstractEPM
from smac.optimizer.acquisition import AbstractAcquisitionFunction


class Direct(AbstractAcquisitionFunction):
    def __init__(self, model: AbstractEPM = None, **kwargs):
        super().__init__(model)
        self.long_name = 'Direct'

    def _compute(self, X: np.ndarray, **kwargs):
        loss, closeness = self.model.predict(X)
        return 1 - (loss + closeness) / 2


class DirectNoDistance(AbstractAcquisitionFunction):
    def __init__(self, model: AbstractEPM = None, **kwargs):
        super().__init__(model)
        self.long_name = 'Direct'

    def _compute(self, X: np.ndarray, **kwargs):
        loss, closeness = self.model.predict(X)
        return 1 - loss
