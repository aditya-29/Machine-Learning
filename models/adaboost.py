from models.BaseModel import Model
import numpy as np

class DecisionStump:
    def __init__(self):
        self.polarity = 1   # this tells us if the sample is classified as 1 or -1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, x):
        n_samples = x.shape[0]
        x_col = x[:, self.feature_idx]
        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[x_col < self.threshold] = -1
        else:
            predictions[x_col > self.threshold] = -1

        return predictions



class AdaBoost(Model):
    def __init__(self, n_clf = 5):
        self.n_clf = n_clf

    def fit(self, x, y):
        # get the number of samples and number of features
        n_samples, n_features = x.shape

        # init the weights
        w = np.full(n_samples, (1/n_samples))

        # iterate through all the classifiers
        self.clfs = []
        
        # for every classifier
        for _ in range(self.n_clf):
            # create a decision stump
            clf = DecisionStump()
            # assign the default min error = "inf"
            min_error = float("inf")
            # iterate through every feature in the training data
            for feature_i in range(n_features):
                # get only that specific feature
                x_col = x[:, feature_i]
                # get the thresholds which are unique values for the current feature
                thresholds = np.unique(x_col)
                # iterate through every thresholds
                for thresh in thresholds:
                    # set the polarity to 1
                    p = 1
                    # set default predictions to one
                    predictions = np.ones(n_samples)
                    # swap the sign if the current value if less than threshold
                    predictions[x_col < thresh] = -1

                    # calculate the number of misclassified data
                    misclassified = w[y != predictions]
                    # calculate the error
                    error = sum(misclassified)

                    # if the error is greater than 0.5 then swap 
                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # if the min error is found then store it
                    if error < min_error:
                        min_error = error
                        clf.polarity = p
                        clf.threshold = thresh
                        clf.feature_idx = feature_i

            # calculate the alpha  (prediciton power)
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1-error) / (error + EPS))

            # calculate the predictons
            predictions = clf.predict(x)
            # update the weights
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            # add the current classifier to the list of classifiers
            self.clfs.append(clf)

    def predict(self, x):
        clf_preds = [clf.alpha * clf.predict(x) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred

            
            
