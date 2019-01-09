"""Modelling utility functions and classes"""

import time
import datetime
import pickle
import logging
import numpy as np
import pandas as pd
import titanic.tools as tools

from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from titanic.config import PROJECT_ROOT_ABS_PATH


class SimpleDataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, median_cols=None, mode_cols=None):
        self.median_cols = median_cols
        self.median = None
        self.mode_cols = mode_cols
        self.mode = None

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        if self.median_cols is not None:
            self.median = X[self.median_cols].median()
        if self.mode_cols is not None:
            self.mode = X[self.mode_cols].mode().iloc[0, :]
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X = X.copy()
        if self.median_cols is not None:
            X.fillna(self.median, inplace=True)
        if self.mode_cols is not None:
            X.loc[:, self.mode_cols] = X.loc[:, self.mode_cols].fillna(self.mode)
        return X


class DataFrameDummifier(BaseEstimator, TransformerMixin):
    def __init__(self, params={}):
        self.params = params
        self.cols = None

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X = X.copy()
        X = pd.get_dummies(X, **self.params)
        return X


class CrossValidatedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clf):
        self.clf = clf
        self.cvs_history = dict()
        self.init_params = clf.get_params()
        self.best_params = None
        self._params_strategy = 'init'
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        predicted = self.clf.predict(X)
        return predicted

    def _print_cvs_history(self, depth=3):
        key = max(self.cvs_history)
        for r in range(depth):
            if key == 0:
                break
            print(f"RUN {key} score: {self.cvs_history[key]['score']}")
            key = key - 1

    def cross_val_score(self, X, y, scoring='accuracy', hprint=True, hprint_depth=3, **kwargs):
        start_time = time.time()
        score = cross_val_score(self.clf, X, y, scoring=scoring, **kwargs).mean()
        end_time = time.time()
        if self.cvs_history == dict():
            key = 1
        else:
            key = max(self.cvs_history) + 1
        self.cvs_history[key] = dict()
        self.cvs_history[key]['timestamp'] = (datetime.datetime
                                              .fromtimestamp(start_time)
                                              .strftime('%Y-%m-%d %H:%M:%S'))
        self.cvs_history[key]['executed_in'] = end_time - start_time
        self.cvs_history[key]['score'] = score
        if hprint:
            self._print_cvs_history(depth=hprint_depth)
        return score

    def grid_search_cv(self, X, y, param_grid, **kwargs):
        grid = GridSearchCV(self.clf, param_grid=param_grid, **kwargs)
        grid.fit(X, y)
        self.best_params = grid.best_params_
        print('best_params:')
        print(grid.best_params_)
        print('best_score:')
        print(grid.best_score_)
        return grid.best_params_, grid.best_score_

    @property
    def params_strategy(self):
        return self._params_strategy

    @params_strategy.setter
    def params_strategy(self, params_strategy):
        if params_strategy == 'init':
            self.clf.set_params(**self.init_params)
            self._params_strategy = 'init'
        elif params_strategy == 'best':
            self.clf.set_params(**self.best_params)
            self._params_strategy = 'best'
        else:
            raise ValueError("Attribute 'params_strategy' must be either 'init' or 'best'.")

    def serialize(self, file_path):
        tools.serialize(self, file_path=file_path)

    @classmethod
    def train(cls, clf, X, y, param_grid=None, param_strategy='init', logdir_name=None, serialize_to=None):
        model = cls(clf)
        if param_grid is not None:
            model.grid_search_cv(X, y, param_grid)
            model.params_strategy = param_strategy
        model.cross_val_score(X, y)
        if serialize_to is not None:
            model.serialize(serialize_to)
        if logdir_name is not None:
            logdir_path = PROJECT_ROOT_ABS_PATH / 'logs' / 'models' / logdir_name
            logdir_path.mkdir(parents=True, exist_ok=True)
            logfile_name = (model.cvs_history[1]['timestamp'] + r'.log').replace(r':', r'-')
            file_handler = logging.FileHandler(logdir_path / logfile_name, mode='w')
            model.logger.addHandler(file_handler)
            model.logger.setLevel(logging.DEBUG)
            model.logger.info(model.cvs_history)
            file_handler.close()
            model.logger.removeHandler(file_handler)
        return model


if __name__ == '__main__':
    with open(r'../../data/processed/X_train.pickle', 'rb') as f:
        X_train = pickle.load(f)
    with open(r'../../data/processed/y_train.pickle', 'rb') as f:
        y_train = pickle.load(f)

    pipe = make_pipeline(
                         SimpleDataFrameImputer(median_cols=['Age'],
                                                mode_cols=['Embarked']),
                         DataFrameDummifier(),
                         LogisticRegression()
                         )
#    logreg = CrossValidatedClassifier(pipe)
#    y_train_predicted = logreg.fit(X_train, y_train).predict(X_train)
#    print(y_train_predicted[:10])
#    param_grid = {'clf__logisticregression__C': [0.5, 1, 2]}
#    grid = GridSearchCV(logreg, param_grid)
#    grid.fit(X_train, y_train)
#    print(grid.best_params_, grid.best_score_)
#    print(logreg.cross_val_score(X_train, y_train))
#    logreg.cross_val_score(X_train, y_train)
    param_grid = {'logisticregression__C': [0.8, 1, 1.2]}
#    logreg.grid_search_cv(X_train, y_train, param_grid)
    logreg = CrossValidatedClassifier.train(pipe, X_train, y_train, param_grid, logdir_name='logreg')





















