"""Modelling utility functions and classes"""

import sys
import time
import datetime
import pickle
import logging
import numpy as np
import pandas as pd
import titanic.tools as tools

from pathlib import Path
from pprint import pformat
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from titanic.tools import print_header


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


class ExtendedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clf):
        self.clf = clf
        self.profile = dict()
        self.init_params = clf.get_params()
        self.best_params = None
        self._params_strategy = 'init'
        if isinstance(clf, Pipeline):
            self.last_step_name = clf.steps[-1][1].__class__.__name__
        else:
            self.last_step_name = clf.__class__.__name__

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        predicted = self.clf.predict(X)
        return predicted

    def cross_val_score(self, X, y, scoring='accuracy', print_cvs=True, **kwargs):
        start_time = time.time()
        score = cross_val_score(self.clf, X, y, scoring=scoring, **kwargs).mean()
        end_time = time.time()
        self.profile['cv_timestamp'] = (datetime.datetime
                                        .fromtimestamp(start_time)
                                        .strftime('%Y-%m-%d %H:%M:%S'))
        self.profile['cv_executed_in'] = end_time - start_time
        self.profile['cv_scoring'] = scoring
        self.profile['cv_score'] = score
        self.profile['params_strategy'] = self._params_strategy
        self.profile['clf_params'] = self.clf.get_params()
        if print_cvs:
            print('cross-validation ', scoring, ': ', self.profile['cv_score'], sep='')
        return score

    def grid_search_cv(self, X, y, param_grid, scoring='accuracy', print_gscv=True, **kwargs):
        grid = GridSearchCV(self.clf, param_grid, scoring=scoring, **kwargs)
        start_time = time.time()
        grid.fit(X, y)
        end_time = time.time()
        self.best_params = grid.best_params_
        self.profile['gs_best_params'] = grid.best_params_
        self.profile['gs_executed_in'] = end_time - start_time
        if print_gscv:
            logger = logging.getLogger(__name__ + '_' +
                                       self.__class__.__name__ +
                                       '_grid_search_cv')
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(handler)
            logger.info('grid search best_params:\n' + str(grid.best_params_))
            logger.info('grid search best ' + scoring + ': ' + str(grid.best_score_))
            handler.close()
            logger.removeHandler(handler)
        return grid.best_params_, grid.best_score_

    @property
    def params_strategy(self):
        return self._params_strategy

    @params_strategy.setter
    def params_strategy(self, params_strategy):
        if params_strategy == 'init':
            self.clf = clone(self.clf).set_params(**self.init_params)
            self._params_strategy = 'init'
        elif params_strategy == 'best':
            self.clf = clone(self.clf).set_params(**self.best_params)
            self._params_strategy = 'best'
        else:
            raise ValueError("Attribute 'params_strategy' must be either 'init' or 'best'.")

    def serialize(self, file_path):
        tools.serialize(self, file_path=file_path)

    @classmethod
    def deserialize(cls, file_path):
        obj = tools.deserialize(file_path=file_path)
        if isinstance(obj, cls):
            return obj
        else:
            raise TypeError(f'Deserialized object must be an instance of class {cls.__name__}')

    def _dump_profile_to_log(self, logdir_path):
        logger = logging.getLogger(__name__ + '_' + self.__class__.__name__)
        logdir_path = Path(logdir_path)
        logdir_path.mkdir(parents=True, exist_ok=True)
        logfile_name = (self.profile['cv_timestamp'] + '_' +
                        self.profile['cv_scoring'] + '_' +
                        str(self.profile['cv_score']) +
                        r'.log').replace(r':', r'-').replace(' ', '_')
        file_handler = logging.FileHandler(logdir_path / logfile_name, mode='w')
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        logger.info(pformat(self.profile))
        file_handler.close()
        logger.removeHandler(file_handler)

    @classmethod
    def cross_validate(cls, clf, X, y, param_grid=None, param_strategy='init',
                       print_cvs=True, print_gscv=True,
                       logdir_path=None, serialize_to=None, sklearn_gscv_kws=None,
                       sklearn_cvs_kws=None):

        # Counter mutable default kwargs
        if sklearn_gscv_kws is None:
            sklearn_gscv_kws = dict()
        if sklearn_cvs_kws is None:
            sklearn_cvs_kws = dict()

        # Train model
        model = cls(clf)
        print_header(model.last_step_name)
        if param_grid is not None:
            model.grid_search_cv(X, y, param_grid, print_gscv=print_gscv, **sklearn_gscv_kws)
            model.params_strategy = param_strategy
        model.cross_val_score(X, y, print_cvs=print_cvs, **sklearn_cvs_kws)

        # Serialize model
        if serialize_to is not None:
            model.serialize(serialize_to)

        # Write log
        if logdir_path is not None:
            model._dump_profile_to_log(logdir_path)
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
                         LogisticRegression(solver='liblinear')
                         )
#    logreg = ExtendedClassifier(pipe)
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
    logreg = ExtendedClassifier.cross_validate(pipe, X_train, y_train, param_grid,
                                               sklearn_gscv_kws={'cv': 3},
                                               sklearn_cvs_kws={'cv': 5},
                                               logdir_path=r'../../logs/models/logreg',
                                               param_strategy='best')
#    pups = ExtendedClassifier.deserialize(r'../../models/logreg.pickle')
#    print(type(pups))
#    tools.serialize(ExtendedClassifier(pipe), '../../models/logreg.pickle')





















