import pytest
import os
import pandas as pd
import titanic.scripts.make_features as ft
from pathlib import Path

DATA_PATH = Path(__file__).parents[1].joinpath(r'data/scripts/make_features')


@pytest.fixture
def titanic_df():
    df = pd.read_excel(DATA_PATH/'make_features.xlsx', sheet_name='input')
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


def test_transform_object_to_categorical():
    train = pd.DataFrame({'a': [1, 2, 3, 4], 'b': ['zero', 'one', 'two', 'three']})
    test = pd.DataFrame({'a': [5, 6, 7, 8], 'b': ['one', 'two', 'four', 'nine']})
    ft.transform_object_to_categorical(train, test)
    assert pd.api.types.is_numeric_dtype(train['a'])
    assert pd.api.types.is_numeric_dtype(test['a'])
    assert pd.api.types.is_categorical_dtype(train['b'])
    assert pd.api.types.is_categorical_dtype(test['b'])
    train_cats = sorted(list(train['b'].cat.categories))
    test_cats = sorted(list(test['b'].cat.categories))
    expected_cats = ['four', 'nine', 'one', 'three', 'two', 'zero']
    assert train_cats == expected_cats
    assert test_cats == expected_cats
