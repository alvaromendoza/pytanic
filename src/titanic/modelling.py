"""Modelling utility functions and classes."""

import io
import sys
import time
import datetime
import logging
from pathlib import Path
from pprint import pformat

import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import titanic.tools as tools


def transform_object_to_categorical(train_df, test_df):
    """For columns with dtype 'object' change dtype to pandas CategoricalDtype in place.

    Assign a common exhaustive set of categories for each feature of type object in
    train and test. The purpose is to eliminate the need to align training and validation/test
    datasets after dummification.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset.
    test_df : pandas.DataFrame
        Test dataset.

    """
    assert np.sum(train_df.columns != test_df.columns) == 0
    obj_cols = list(train_df.select_dtypes(include=np.object).columns)
    for df in [train_df, test_df]:
        for col in obj_cols:
            categories = (set(train_df[col].unique()) |
                          set(test_df[col].unique())) - set([np.nan])
            cat_type = CategoricalDtype(
                categories=categories, ordered=None)
            df[col] = df[col].astype(cat_type)


class CategoricalToString(BaseEstimator, TransformerMixin):
    """Transform columns of dtype 'category' to dtype 'object' ('str').

    Works only with pandas DataFrames. Inherits from scikit-learn base classes and
    is compatible with its API.

    """

    def fit(self, X, y=None):
        """Dummy fit method required for fit_transform to work."""
        return self

    def transform(self, X):
        """Transform cateforical columns in X to string.

        Parameters
        ----------
        X : pandas.DataFrame
            Dataset to be transformed.

        Returns
        -------
        X : pandas.DataFrame
            Transformed copy of input X.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be an instance of pandas.DataFrame.')
        X = X.copy()
        for c in X.select_dtypes(['category']).columns:
            X[c] = X[c].astype(str)
        return X


class SimpleDataFrameImputer(BaseEstimator, TransformerMixin):
    """Impute missing values based on medians or modes of corresponding columns in fitted dataset.

    Works only with pandas DataFrames. Inherits from scikit-learn base classes and
    is compatible with its API.

    Parameters
    ----------
    median_cols : list
        Columns to be imputed with median.
    mode_cols : list
        Columns to be imputed with mode.

    """

    def __init__(self, median_cols=None, mode_cols=None):
        self.median_cols = median_cols
        self.median = None
        self.mode_cols = mode_cols
        self.mode = None

    def fit(self, X, y=None):
        """Fit imputer on X.

        Parameters
        ----------
        X : pandas.DataFrame
            Input dataset from which the imputer learns medians and modes.
        y : None
            Dummy parameter required by scikit-learn.

        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be an instance of pandas.DataFrame.')
        if self.median_cols is not None:
            self.median = X[self.median_cols].median()
        if self.mode_cols is not None:
            self.mode = X[self.mode_cols].mode().iloc[0, :]
        return self

    def transform(self, X):
        """Impute missing values in X.

        Parameters
        ----------
        X : pandas.DataFrame
            Dataset to be imputed.

        Returns
        -------
        X : pandas.DataFrame
            Transformed copy of input X.

        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be an instance of pandas.DataFrame.')
        X = X.copy()
        if self.median_cols is not None:
            X.fillna(self.median, inplace=True)
        if self.mode_cols is not None:
            X.loc[:, self.mode_cols] = X.loc[:, self.mode_cols].fillna(self.mode)
        return X


class DataFrameDummifier(BaseEstimator, TransformerMixin):
    """Transform categorical features to dummy variables.

    Works only with pandas DataFrames. Inherits from scikit-learn base classes and
    is compatible with its API.

    Parameters
    ----------
    get_dummies_kws : dict, optional
        Keyword arguments of pandas.get_dummies. Warning: mutable default value.

    """
    def __init__(self, get_dummies_kws={}):
        self.get_dummies_kws = get_dummies_kws

    def fit(self, X, y=None):
        """Dummy fit method required for fit_transform to work."""
        return self

    def transform(self, X):
        """Transform categorical features to dummy variables in X.

        Parameters
        ----------
        X : pandas.DataFrame
            Dataset to be transformed.

        Returns
        -------
        X : pandas.DataFrame
            Transformed copy of input X.

        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be an instance of pandas.DataFrame.')
        X = X.copy()
        X = pd.get_dummies(X, **self.get_dummies_kws)
        return X


class ExtendedClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper class for scikip-learn classfiers with additional functionality.

    Parameters
    ----------
    clf : instance of scikit-learn classifier or Pipeline

    Attirbutes
    ----------
    init_params : dict
        Initial parameters of clf obtained via clf.get_params()
    best_params : dict
        Best parameters calculated by grid_search_cv.
    _params_strategy : {'init', 'best'}, default 'init'
        Set of parameters used by clf.
    profile : dict
        Info on the last run of cross_val_score and grid_search_cv.
    last_step_name : str
        Name of clf or last step of clf if it is a pipeline.

    """

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
        """Simple wrapper for clf.fit."""
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        """Simple wrapper for clf.predict."""
        predicted = self.clf.predict(X)
        return predicted

    def predict_proba(self, X):
        """Simple wrapper for clf.predict_proba."""
        predicted_proba = self.clf.predict_proba(X)
        return predicted_proba

    def cross_val_score(self, X, y, scoring='accuracy', print_cvs=True, **kwargs):
        """Run scikti-learn's cross_val_score and write results to profile and stdout.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.
        scoring : string, optional, default 'accuracy'
            scikti-learn's cross_val_score parameter of the same name.
        print_cvs : bool, optional, default True
            Print info to stdout.
        kwargs : dict, optional
            Keyword arguments of scikti-learn's cross_val_score.

        Returns
        -------
        score : float
            Mean cross-validated score.

        """
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
        self.profile['X_hash'] = pd.util.hash_pandas_object(X).sum()
        self.profile['y_hash'] = pd.util.hash_pandas_object(y).sum()
        buf = io.StringIO()
        X.info(buf=buf)
        self.profile['X_info'] = buf.getvalue()
        self.profile['y_describe'] = y.describe()
        if print_cvs:
            print('cross-validation ', scoring, ': ', self.profile['cv_score'], sep='')
        return score

    def grid_search_cv(self, X, y, param_grid, scoring='accuracy', print_gscv=True, **kwargs):
        """Run scikti-learn's GridSearchCV and write results to profile and stdout.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector for GridSearchCV.fit, where n_samples is the number
            of samples and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector for GridSearchCV.fit relative to X.
        param_grid : dict or list of dictionaries
            scikti-learn's GridSearchCV parameter of the same name.
        scoring : string, optional, default 'accuracy'
            scikti-learn's GridSearchCV parameter of the same name.
        print_gscv : bool, optional, default True
            Print info to stdout.
        kwargs : dict
            Keyword arguments of scikti-learn's GridSearchCV.

        Returns
        -------
        grid.best_params : dict
            Parameter setting that gave the best results on the hold out data.
        grid.best_score_ : dict
            Mean cross-validated score.
        """
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
        """Getter of the 'private' attribute _params_strategy."""
        return self._params_strategy

    @params_strategy.setter
    def params_strategy(self, params_strategy):
        """Switch parameters of clf from 'init' to 'best' and vice versa.

        Also acts os a setter of the 'private' attribute _params_strategy.

        Parameters
        ----------
        params_strategy : {'init', 'best'}

        """
        if params_strategy == 'init':
            self.clf = clone(self.clf).set_params(**self.init_params)
            self._params_strategy = 'init'
        elif params_strategy == 'best':
            self.clf = clone(self.clf).set_params(**self.best_params)
            self._params_strategy = 'best'
        else:
            raise ValueError("Attribute 'params_strategy' must be either 'init' or 'best'.")

    def serialize(self, file_path):
        """Simple wrapper for tools.serialize."""
        tools.serialize(self, file_path=file_path)

    @classmethod
    def deserialize(cls, file_path):
        """Wrapper for tools.deserialize with membership check."""
        obj = tools.deserialize(file_path=file_path)
        if isinstance(obj, cls):
            return obj
        else:
            raise TypeError(f'Deserialized object must be an instance of class {cls.__name__}')

    def _dump_profile_to_log(self, logdir_path):
        """Write contents of profile attribute to log file.

        Parameters
        ----------
        logdir_path : str
            Path to directory where log file is to be created.

        """
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
        """Instantiate model, grid search parameters, perform cross-validation, serialize model.

        Parameters
        ----------
        clf : instance of scikit-learn classifier or Pipeline
            Classifier to be instantiated.
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.
        print_cvs : bool, optional, default True
            Print cross-validation info to stdout.
        print_gscv : bool, optional, default True
            Print grid search info to stdout.
        sklearn_cvs_kws : dict, optional
            Keyword arguments of scikti-learn's cross_val_score.
        param_grid : dict or list of dictionaries
            scikti-learn's GridSearchCV parameter of the same name.
            If None, no grid search is performed.
        sklearn_gscv_kws : dict
            Keyword arguments of scikti-learn's GridSearchCV.
        logdir_path : str, optional
            Path to directory where log file is to be created.
            If None, no log file is created.
        serialize_to : str, pathlib.Path, optional
            Path to pickle file into which cross-validated clf will be dumped.
            If None, the model is not  serilized.

        Returns
        -------
        model : instance of ExtendedClassifier
            Cross-validated model.

        """
        # Counter mutable default kwargs
        if sklearn_gscv_kws is None:
            sklearn_gscv_kws = dict()
        if sklearn_cvs_kws is None:
            sklearn_cvs_kws = dict()

        # Train model
        model = cls(clf)
        tools.print_header(model.last_step_name)
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
