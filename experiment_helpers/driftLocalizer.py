from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.interpolate import CubicHermiteSpline
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV
import numpy as np


# Fully integrated localizer
class Localizer(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.05, min_samples_leaf=None, smoothing=0.00001):
        assert 0 < smoothing < 0.5
        self.localizer_model = None
        self.alpha = alpha
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
        self.transform_function = None

    def spline_max(self, x, y, n, sample=1000):
        dy = np.array([0, (y[2] - y[0]) / (x[2] - x[0]) / n, 0, (y[4] - y[2]) / (x[4] - x[2]) / n, 0])
        spline = CubicHermiteSpline(x, y, dy)
        return ((np.exp(spline(np.linspace(0, 1, 1000))) - 0.5) / (0.5 - self.smoothing) * 0.5 + 0.5).max(), spline

    def find_spline(self, x, y, sample=1000):
        lo, hi = 0.001, 2
        while True:
            mid = (lo + hi) / 2
            vmid, msp = self.spline_max(x, y, mid, sample)
            if vmid > 1:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-5:
                return msp

    def fit(self, X, y):
        assert np.logical_or(y == 0, y == 1).all()

        if self.min_samples_leaf is None:
            print("Determine optimal parameters using cross validation")
            gs = GridSearchCV(estimator=RandomForestClassifier(n_estimators=100),
                              param_grid={"min_samples_leaf": [20, 30, 50, 100]}, cv=5, scoring='neg_log_loss',
                              n_jobs=-1).fit(X, y)
            self.min_samples_leaf = gs.best_params_["min_samples_leaf"]

        mean = y.mean()
        dt_lo = stats.binom.ppf(self.alpha, self.min_samples_leaf, mean) / self.min_samples_leaf
        dt_hi = stats.binom.ppf(1 - self.alpha, self.min_samples_leaf, mean) / self.min_samples_leaf
        print(f"low threshold: {dt_lo} Mean:{mean} High threshold:{dt_hi}, No. Leaves:{self.min_samples_leaf}")
        try:
            spline = self.find_spline(np.array([0, dt_lo, mean, dt_hi, 1]), np.array(
                [np.log(self.smoothing), np.log(0.5), np.log(1 - self.smoothing), np.log(0.5), np.log(self.smoothing)]))
        except:
            raise ValueError(f"threshold error: {dt_lo} {mean} {dt_hi}, {self.min_samples_leaf}")

        self.transform_function = lambda prob_pred: (np.exp(spline(prob_pred)) - 0.5) / (
                    0.5 - self.smoothing) * 0.5 + 0.5

        self.localizer_model = RandomForestClassifier(min_samples_leaf=self.min_samples_leaf).fit(X, y)
        return self

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, X):
        prob_pred = self.localizer_model.predict_proba(X)[:, 1]
        c2 = self.transform_function(self.localizer_model.predict_proba(X)[:, 1])
        c0 = np.ones(c2.shape[0]) - (prob_pred < 0.5) * c2 - (prob_pred >= 0.5)
        c1 = np.ones(c2.shape[0]) - (prob_pred > 0.5) * c2 - (prob_pred <= 0.5)

        return np.array([c0, c1, c2]).T

    def l_predict(self, X):
        probabilities = self.l_predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def l_predict_proba(self, X):
        prob_pred = self.localizer_model.predict_proba(X)
        return prob_pred
