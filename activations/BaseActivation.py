from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class Activation:
    def __new__(cls, val: [list, np.ndarray, np.generic] ) -> Any:
        instance = super().__new__(cls)
        return instance.get_vales(val)
    
    @abstractmethod
    def get_vales(self, *args, **kwargs):
        raise NotImplementedError("The get_values() is not implemented.")