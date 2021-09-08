import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class NumericalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.select_dtypes(include=[np.number])
        return X


class CategoricalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.select_dtypes(include=['object'])
        return X
