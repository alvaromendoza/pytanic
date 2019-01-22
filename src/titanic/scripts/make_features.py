"""Perform feature engineering and make training and test datasets out of raw data."""

import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from pandas.api.types import CategoricalDtype
from titanic.tools import serialize


def load_train_test(raw_data_path=r'data/raw'):
    raw_data_path = Path(raw_data_path)
    train = pd.read_csv(raw_data_path.joinpath('train.csv'))
    test = pd.read_csv(raw_data_path.joinpath('test.csv'))
    return train, test


def split_X_y(df, y_col='Survived'):
    y = df[y_col].copy()
    X = df.drop(y_col, axis=1, inplace=False)
    return X, y


def add_title(df):
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'].replace('Mlle', 'Miss', inplace=True)
    df['Title'].replace('Ms', 'Miss', inplace=True)
    df['Title'].replace('Mme', 'Mrs', inplace=True)
    title_other_filter = ~df['Title'].isin(['Mr', 'Master', 'Mrs', 'Miss'])
    df.loc[title_other_filter, 'Title'] = 'Other'


def add_familysize(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1


def add_deck(df):
    df['Deck'] = df['Cabin'].str.extract(r'([A-Z])+', expand=False)
    df['Deck'].fillna('X', inplace=True)


def recode_sex(df):
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0}).astype(np.int8)
    df.rename({'Sex': 'IsMale'}, axis=1, inplace=True)


def drop_cols(df, cols_to_drop=None):
    if cols_to_drop is None:
        cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df.drop(cols_to_drop, axis=1, inplace=True)


def transform_object_to_categorical(train_df, test_df):
    """For columns with dtype 'object' change dtype to pandas CategoricalDtype.

    Assign a common exhaustive set of categories for each feature of type object in
    train and test. The purpose is to eliminate the need to align training and validatio/test
    datasets after dummification.
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


def main():
    train, test = load_train_test()
    print('Train and test datasets loaded.')
    X_train, y_train = split_X_y(train)
    X_test = test
    for df in [X_train, X_test]:
        add_title(df)
        add_familysize(df)
        add_deck(df)
        recode_sex(df)
        drop_cols(df)
    transform_object_to_categorical(X_train, X_test)
    print('Transformations performed.')
    serialize(X_train, r'data/processed/X_train.pickle')
    serialize(X_test, r'data/processed/X_test.pickle')
    serialize(y_train, r'data/processed/y_train.pickle')
    print('DataFrames serialized.')
    return X_train, X_test, y_train


if __name__ == '__main__':
    X_train, X_test, y_train = main()
