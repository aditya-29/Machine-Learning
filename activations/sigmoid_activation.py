from typing import Any
from activations.BaseActivation import Activation
import numpy as np


class Sigmoid(Activation):
    def _sigmoid(self, values):
        return 1 / (1 + np.exp(-values))

    def get_vales(self, val):
        return self._sigmoid(val)
    