import numpy as np  
from models.BaseModel import Model

class NaiveBayes(Model):
    def __init__(self):
        self.mean = None
        self.var = None
        self.classes = None
        

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.mean = np.zeros((n_classes, n_features), dtype = np.float32)
        self.var = np.zeros((n_classes, n_features), dtype=np.float32)
        self.priors = np.zeros(n_classes, dtype = np.float32)

        for idx, c in enumerate(self.classes):
            xc = X[c == y]
            
            self.mean[idx, :] = xc.mean(axis=0)
            self.var[idx, :] = xc.var(axis=0)
            self.priors[idx] = xc.shape[0] / float(n_samples)

    def _pdf(self, class_index, x):
        mean = self.mean[class_index]
        var = self.var[class_index]
        denominator = np.sqrt(2*np.pi*var)
        power = (x - mean)**2 / (2*var)
        return (1/denominator) * np.exp(-power)
    
    def _predict(self, x):
        posteriors = []
        
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred