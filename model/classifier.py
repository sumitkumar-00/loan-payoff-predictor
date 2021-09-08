import os
import sys

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier

from joblib import dump


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


def load_data(file):
    """
    This function loads the csv file and returns a  pandas dataframe
    :param file: path to file containing data
    :return: dataframe
    """

    df = pd.read_csv(file)
    print("{0} observations and {1} features loaded from the csv file".format(df.shape[0], df.shape[1]))
    return df


def preprocessing():
    """
    Builds pipeline and returns a pipeline object
    :return: pipeline
    """
    pipeline_numerical = Pipeline(steps=
                                  [('numerical_features', NumericalFeatures()),
                                   ('standard_scaler', StandardScaler())
                                   ])

    pipeline_categorical = Pipeline(steps=
                                    [('categorical_features', CategoricalFeatures()),
                                     ('OneHotEncoder', OneHotEncoder())
                                     ]
                                    )
    pipeline_preprocessing = FeatureUnion(
        [('numerical', pipeline_numerical),
         ('categorical', pipeline_categorical)
         ]
    )

    return pipeline_preprocessing


def check_model_accuracy(name, model, param_grid, X_train, y_train, X_test, y_test):
    pipeline = Pipeline(steps=
                        [('preprocessing', preprocessing()),
                         (name, model)
                         ])

    gs = GridSearchCV(pipeline, param_grid=param_grid, cv=3, scoring='roc_auc', verbose=0, n_jobs=-1)
    gs.fit(X_train, y_train)
    y_prob = gs.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)
    print(name + ' auc_score is ' + str(auc_score))
    result = [name, gs.best_estimator_, auc_score]
    return result


def main():
    if len(sys.argv) < 2:
        print("Please provide path to the csv file")
    else:
        file = sys.argv[1]

    try:
        df = load_data(file)
    except FileNotFoundError:
        print("File not found. Please provide correct filepath")

    # random_state to be used for repeatability
    random_state = 7

    X = df.drop(['not.fully.paid'], axis=1)
    y = df['not.fully.paid'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train = y_train.copy()
    y_test = y_test.copy()

    results = []

    # Random Forest Classifier
    param_grid = {'RandomForest__criterion': ['gini', 'entropy'],
                  'RandomForest__n_estimators': np.arange(100, 1000, 300),
                  'RandomForest__random_state': [random_state],
                  'RandomForest__max_depth': range(2, 8, 2)
                  }
    result = check_model_accuracy('RandomForest', RandomForestClassifier(), param_grid, X_train, y_train, X_test,
                                  y_test)
    results.append(result)

    # Linear Regression
    param_grid = {'LogisticRegression__C': np.linspace(0.001, 5, 100),
                  'LogisticRegression__random_state': [random_state]
                  }
    result = check_model_accuracy('LogisticRegression', LogisticRegression(), param_grid, X_train, y_train, X_test,
                                  y_test)
    results.append(result)

    # KNeighbor Classifier
    param_grid = {'KNeighborClassifier__n_neighbors': range(10, 100, 5),
                  'KNeighborClassifier__p': range(1, 3)
                  }
    result = check_model_accuracy('KNeighborClassifier', KNeighborsClassifier(), param_grid, X_train, y_train, X_test,
                                  y_test)
    results.append(result)

    # SGDClassifier
    param_grid = {
        'SGDClassifier__loss': ['hinge', 'log', 'perceptron'],
        'SGDClassifier__random_state': [random_state],
        'SGDClassifier__penalty': ['l1', 'l2', 'elasticnet'],
        'SGDClassifier__alpha': np.linspace(0.0001, 1, 100)
    }
    result = check_model_accuracy('SGDClassifier', SGDClassifier(), param_grid, X_train, y_train, X_test, y_test)
    results.append(result)

    # XGBClassifier
    param_grid = {
        'XGBClassifier__learning_rate': np.linspace(0.01, 1, 5)
    }
    result = check_model_accuracy('XGBClassifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid, X_train, y_train,
                                  X_test, y_test)
    results.append(result)

    # GradientBoostingClassifier
    param_grid = {
        'GradientBoostingClassifier__learning_rate': np.linspace(0.1, 1, 25),
        'GradientBoostingClassifier__max_depth': [3, 4, 5],
        'GradientBoostingClassifier__random_state': [random_state]
    }
    result = check_model_accuracy('GradientBoostingClassifier', GradientBoostingClassifier(), param_grid, X_train,
                                  y_train, X_test, y_test)
    results.append(result)

    # Conclusion
    conclusion(results)


def conclusion(results):
    auc_scores = [result[2] for result in results]
    best_model_index = auc_scores.index(max(auc_scores))
    print(results[best_model_index][0] + " is the best performing model with auc_score of " + str(max(auc_scores)))
    dump(results[best_model_index][1], 'model/loan_data.pkl')
    print("Saving best model as a pickled file")


if __name__ == "__main__":
    main()
