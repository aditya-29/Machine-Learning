from loss.BaseLoss import Loss
import numpy as np


class Categorical_CrossEntropy(Loss):
    def _cat_loss(self, preds, actuals, **kwargs):
        return -np.sum(actuals * np.log(preds + 10**-100))


    def get_loss(self, preds, actuals, **kwargs):
        self._cat_loss(preds, actuals, **kwargs)
