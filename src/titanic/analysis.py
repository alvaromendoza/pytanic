"""Analytical utility functions."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def get_nan_counts(df):
    """Calculate count and percentage of NaN in each column of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Resulting DataFrame with columns 'Feature', 'NaN count', and 'NaN percentage'.

    """
    nan_counts = pd.DataFrame(df.isnull().sum()).reset_index()
    nan_counts.columns = ['Feature', 'NaN count']
    nan_counts['NaN percentage'] = nan_counts['NaN count'] / df.shape[0]
    nan_counts.sort_values('NaN percentage', inplace=True, ascending=False)
    nan_counts.reset_index(inplace=True, drop=True)
    return nan_counts


def get_count_percentage(df, column, sort='count'):
    """Calculate value counts and corresponding percentages for a given column in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    column : str
        Input column name.
    sort : {'count', 'name'}
        Method of sorting the resulting DataFrame
        ('count' - by value count, 'name' - by column name).

    Returns
    -------
    pandas.DataFrame
        Resulting DataFrame with columns {column}, 'Count', and 'Percentage'.

    """
    result = df[column].value_counts().reset_index()
    result.columns = [column, 'Count']
    result['Percentage'] = result['Count'] / result['Count'].sum()
    if sort == 'count':
        sort_by = 'Count'
        is_ascending = False
    elif sort == 'name':
        sort_by = column
        is_ascending = True
    else:
        raise ValueError("Parameter 'sort' must be either 'count' or 'name'.")
    result.sort_values(sort_by, inplace=True, ascending=is_ascending)
    result.reset_index(drop=True, inplace=True)
    return result


def association_test(X, y, sort_column=2):
    """Create a DataFrame with measures of association between features in X, and y.

    Measures of association include 'Pearson's r', 'Spearman's rho', 'Root R^2',
    and 'p-value of F'.

    Parameters
    ----------
    X : array_like
        pandas.DaraFrame or another oject that can be converted to one.
        May contain categorical features of type object.
    y : array_like, shape (X.shape[0], 0 or 1)
        pandas.Series or another oject that can be converted into one.
    sort_column : int or str literal'index', default 2 ('Root R^2')
        Index of column used to sort the resulting DataFrame in descending order.
        If set to string 'index', the resulting DataFrame is sorted using column names of X.

    Returns
    -------
    pandas.DataFrame

    """
    # Regression
    output_regr = []
    X = pd.DataFrame(X)
    for feature in X.columns:
        if X[feature].dtype == np.object:
            dm = pd.get_dummies(X[feature])
        else:
            dm = X[feature]
        dm = sm.add_constant(dm)
        result = sm.OLS(y, dm.astype(float), missing='drop').fit()
        output_regr.append({'Feature': feature, 'Root R^2': np.sqrt(
            result.rsquared), 'p-value of F': result.f_pvalue})
    output_regr = pd.DataFrame(output_regr).set_index('Feature')
    output_regr.index.name = None

    # Correlation
    X = X.select_dtypes(exclude=np.object)
    pearson = X.apply(lambda col: col.corr(y, method='pearson'))
    spearman = X.apply(lambda col: col.corr(y, method='spearman'))
    output_correl = pd.concat([pearson, spearman], axis=1)
    output_correl.columns = ["Pearson's r", "Spearman's rho"]

    # Combined output
    sort = True if sort_column == 'index' else False
    output = pd.concat([output_correl, output_regr], sort=sort, axis=1)
    if sort_column != 'index':
        output = output.sort_values(output.columns[sort_column], ascending=False)
    return output
