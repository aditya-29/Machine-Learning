from loss.BaseLoss import Loss
import numpy as np


class MeanSquaredErrorLoss(Loss):
    def _mse_loss(self, preds, actuals, **kwargs):
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        if not isinstance(actuals, np.ndarray):
            actuals = np.array(actuals)

        return np.sum((preds - actuals)**2) / preds.shape[0]


    def get_loss(self, preds, actuals, **kwargs):
        self._mse_loss(preds, actuals, **kwargs)
