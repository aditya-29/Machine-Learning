from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class Loss:
    def __new__(cls, preds: [int, float, list, np.ndarray, np.generic],  actuals: [int, float, list, np.ndarray, np.generic] ) -> Any:
        instance = super().__new__(cls)
        return instance.get_loss(preds, actuals)
    
    @abstractmethod
    def get_loss(self, preds, actuals, **kwargs):
        raise NotImplementedError("The get_loss() is not implemented.")