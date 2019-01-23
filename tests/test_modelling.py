"""Tests for module 'modelling'."""

import os
import random as rn
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV

from titanic.modelling import transform_object_to_categorical
from titanic.modelling import CategoricalToString, SimpleDataFrameImputer, DataFrameDummifier
from titanic.modelling import ExtendedClassifier
from titanic.config import RANDOM_SEED

np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = '0'
rn.seed(RANDOM_SEED)

DATA_PATH = Path(__file__).parent.joinpath(r'data')


@pytest.fixture
def X_raw():
    df = pd.read_excel(DATA_PATH/'modelling.xlsx', sheet_name='X_raw')
    return df


@pytest.fixture
def X():
    df = pd.read_excel(DATA_PATH/'modelling.xlsx', sheet_name='X_processed')
    return df


@pytest.fixture
def y():
    df = pd.read_excel(DATA_PATH/'modelling.xlsx', sheet_name='y')
    return df['Survived']


def test_transform_object_to_categorical():
    train = pd.DataFrame({'a': [1, 2, 3, 4], 'b': ['zero', 'one', 'two', 'three']})
    test = pd.DataFrame({'a': [5, 6, 7, 8], 'b': ['one', 'two', 'four', 'nine']})
    transform_object_to_categorical(train, test)
    assert pd.api.types.is_numeric_dtype(train['a'])
    assert pd.api.types.is_numeric_dtype(test['a'])
    assert pd.api.types.is_categorical_dtype(train['b'])
    assert pd.api.types.is_categorical_dtype(test['b'])
    train_cats = sorted(list(train['b'].cat.categories))
    test_cats = sorted(list(test['b'].cat.categories))
    expected_cats = ['four', 'nine', 'one', 'three', 'two', 'zero']
    assert train_cats == expected_cats
    assert test_cats == expected_cats


def test_sdfi(X_raw):
    """Test SimpleDataFrameImputer."""
    train = X_raw.iloc[:30, :]
    test = X_raw.iloc[30:, :]
    print('\ntrain.shape:', train.shape)
    print('test.shape:', test.shape)
    imputer = SimpleDataFrameImputer(median_cols=['Age'], mode_cols=['Embarked'])
    imputer.fit(train)
    test_imputed = imputer.transform(test)
    print('test_imputed:\n', test_imputed)
    print('test_imputed.shape:', test_imputed.shape)
    assert (test_imputed.loc[43, 'Age'] == 34) and (test_imputed.loc[45, 'Age'] == 34)
    assert (test_imputed.loc[44, 'Embarked'] == 'S') and (test_imputed.loc[47, 'Embarked'] == 'S')


def test_dfd(X_raw):
    """Test DataFrameDummifier."""
    dummifier = DataFrameDummifier()
    X_raw_dummified = dummifier.transform(X_raw.iloc[:10, :])
    print('\nX_raw_dummified.head:\n', X_raw_dummified)
    assert X_raw_dummified.loc[0, 'Embarked_C'] == 1
    assert X_raw_dummified.loc[2, 'Embarked_S'] == 1
    assert X_raw_dummified.loc[7, 'Embarked_Q'] == 1


def test_cts(X_raw):
    """Test CategoricalToString."""
    X_raw['Embarked'] = X_raw['Embarked'].astype('category')
    print("\nX_raw['Embarked'].dtype:", X_raw['Embarked'].dtype)
    X_raw_string = CategoricalToString().transform(X_raw)
    print("X_raw_string['Embarked'].dtype:", X_raw_string['Embarked'].dtype)
    assert X_raw_string['Embarked'].dtype == 'object'


@pytest.mark.parametrize('clf,name', [
        (LogisticRegression(), 'LogisticRegression'),
        (RandomForestClassifier(), 'RandomForestClassifier'),
        (make_pipeline(StandardScaler(), SVC()), 'SVC')
])
def test_ec_init(clf, name):
    """Test ExtendedClassifier.__init__."""
    model = ExtendedClassifier(clf)
    assert isinstance(model.clf, clf.__class__)
    assert model.profile == {}
    assert model.init_params == clf.get_params()
    assert model.best_params is None
    assert model._params_strategy == 'init'
    assert model.last_step_name == name


def test_ec_cross_val_score(X, y):
    """Test ExtendedClassifier.cross_val_score."""
    clf = LogisticRegression(solver='liblinear')
    model = ExtendedClassifier(clf)
    method_score = model.cross_val_score(X, y, cv=3)
    score = cross_val_score(clf, X, y, cv=3).mean()
    assert method_score == score
    assert model.profile['cv_score'] == method_score


def test_ec_grid_search_cv(X, y):
    """Test ExtendedClassifier.grid_search_cv."""
    clf = SVC(gamma='scale')
    model = ExtendedClassifier(clf)
    param_grid = {'C': [0.01, 0.1, 1, 10]}
    method_best_params, method_best_score = model.grid_search_cv(X, y, param_grid, cv=2)
    grid = GridSearchCV(clf, param_grid, cv=2)
    grid.fit(X, y)
    assert grid.best_params_ == model.profile['gs_best_params']


def test_ec_params_strategy():
    """Test ExtendedClassifier.params_strategy."""
    init_params = {'C': 0.57}
    best_params = {'C': 1.25}
    clf = LogisticRegression(**init_params)
    model = ExtendedClassifier(clf)
    model.best_params = {'C': 1.25}
    model.params_strategy = 'best'
    assert model.clf.get_params()['C'] == best_params['C']
    model.params_strategy = 'init'
    assert model.clf.get_params()['C'] == init_params['C']


def test_ec_dump_profile_to_log(tmpdir):
    """Test ExtendedClassifier._dump_profile_to_log."""
    clf = LogisticRegression()
    model = ExtendedClassifier(clf)
    model.profile['cv_timestamp'] = str(123)
    model.profile['cv_scoring'] = 'abc'
    model.profile['cv_score'] = 0.456
    model._dump_profile_to_log(tmpdir)
    assert tmpdir.join('123_abc_0.456.log').isfile()
