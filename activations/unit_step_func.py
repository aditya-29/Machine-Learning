import numpy as np
from activations.BaseActivation import Activation

class UnitStep(Activation):
    def _unit_step(self, val):
        return np.where(val>=0, 1, 0)
    
    def get_vales(self, val):
        return self._unit_step(val)
    