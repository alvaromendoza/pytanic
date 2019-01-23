"""Tests for script 'make_features.py'."""

import os
from pathlib import Path

import pytest
import pandas as pd

import titanic.scripts.make_features as ft

DATA_PATH = Path(__file__).parent.joinpath(r'data')


@pytest.fixture
def titanic_df():
    df = pd.read_excel(DATA_PATH/'scripts_make_features.xlsx', sheet_name='input')
    return df


def test_load_train_test(tmpdir):
    raw_data_path = tmpdir.mkdir('data').mkdir('raw')
    pd.DataFrame \
        .from_dict({'a': [1, 2, 3], 'b': [4, 5, 6]}) \
        .to_csv(os.path.join(raw_data_path, 'train.csv'))
    pd.DataFrame \
        .from_dict({'c': [7, 8, 9], 'd': [10, 11, 12]}) \
        .to_csv(os.path.join(raw_data_path, 'test.csv'))
    train, test = ft.load_train_test(raw_data_path)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)


def test_split_X_y():
    df = pd.DataFrame.from_dict({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    X, y = ft.split_X_y(df, y_col='b')
    assert all(X.columns == ['a', 'c'])
    assert isinstance(y, pd.Series)
    assert y.name == 'b'


def test_add_title(titanic_df):
    ft.add_title(titanic_df)
    assert all(titanic_df['Title'] == ['Mrs', 'Master', 'Mr', 'Miss', 'Miss', 'Other',
                                       'Mrs', 'Miss', 'Master', 'Other'])


def test_add_familysize(titanic_df):
    ft.add_familysize(titanic_df)
    assert all(titanic_df['FamilySize'] == [3, 7, 1, 1, 1, 1, 3, 1, 3, 1])


def test_add_deck(titanic_df):
    ft.add_deck(titanic_df)
    deck = pd.Series(['B', 'F', 'X', 'C', 'X', 'D', 'X', 'X', 'C', 'X'])
    assert titanic_df['Deck'].equals(deck)


def test_recode_sex(titanic_df):
    ft.recode_sex(titanic_df)
    assert all(titanic_df['IsMale'] == [0, 1, 1, 0, 0, 1, 0, 0, 1, 0])


def test_drop_cols(titanic_df):
    cols_to_drop = ['Fare', 'SibSp']
    ft.drop_cols(titanic_df, cols_to_drop)
    assert all([c not in titanic_df.columns for c in cols_to_drop])
