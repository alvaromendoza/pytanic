import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from titanic.analysis import get_nan_counts, get_count_percentage, association_test

DATA_PATH = Path(__file__).parent.joinpath(r'data/analysis')


def read_small_xlsx(io, sheet_name=None, print_info=True, **kwargs):
    io = Path(io)
    df = pd.read_excel(io, sheet_name=sheet_name, **kwargs)
    if print_info:
        print(io.name, sheet_name, '\n')
        print(df, '\n')
        print(io.name, sheet_name, 'dtypes', '\n')
        print(df.dtypes, '\n')
    return df


dfs = dict()
dfs['gnc'] = dict()
dfs['gnc']['inp'] = read_small_xlsx(DATA_PATH/'get_nan_counts.xlsx', sheet_name='input')
dfs['gnc']['exp'] = read_small_xlsx(DATA_PATH/'get_nan_counts.xlsx', sheet_name='expected')


def test_get_nan_counts():
    nan_counts = get_nan_counts(dfs['gnc']['inp'])
    assert all(nan_counts['Feature'] == dfs['gnc']['exp']['Feature'])
    assert all(nan_counts['NaN count'] == dfs['gnc']['exp']['NaN count'])
    assert all(np.isclose(nan_counts['NaN percentage'], dfs['gnc']['exp']['NaN percentage']))


dfs['gcp'] = dict()
dfs['gcp']['inp'] = read_small_xlsx(DATA_PATH/'get_count_percentage.xlsx', sheet_name='input')
dfs['gcp']['exp_a'] = read_small_xlsx(DATA_PATH/'get_count_percentage.xlsx',
                                      sheet_name='expected_a')
dfs['gcp']['exp_b'] = read_small_xlsx(DATA_PATH/'get_count_percentage.xlsx',
                                      sheet_name='expected_b')


def test_get_count_percentage_sort_by_count():
    cnt_pct = get_count_percentage(dfs['gcp']['inp'], 'a', sort='count')
    assert all(cnt_pct['a'] == dfs['gcp']['exp_a']['a'])
    assert all(cnt_pct['Count'] == dfs['gcp']['exp_a']['Count'])
    assert all(cnt_pct['Percentage'] == dfs['gcp']['exp_a']['Percentage'])


def test_get_count_percentage_sort_by_name():
    cnt_pct = get_count_percentage(dfs['gcp']['inp'], 'b', sort='name')
    assert all(cnt_pct['b'] == dfs['gcp']['exp_b']['b'])
    assert all(cnt_pct['Count'] == dfs['gcp']['exp_b']['Count'])
    assert all(cnt_pct['Percentage'] == dfs['gcp']['exp_b']['Percentage'])


def test_get_count_percentage_raises_valueerror():
    with pytest.raises(ValueError):
        get_count_percentage(dfs['gcp']['inp'], 'b', sort='splunge')


dfs['at'] = dict()
dfs['at']['inp'] = read_small_xlsx(DATA_PATH/'association_test.xlsx', sheet_name='input')


def test_association_test():
    at = association_test(dfs['at']['inp'].iloc[:, 1:4], dfs['at']['inp']['Pigs'])
    assert np.isclose(at.loc['Gas', "Pearson's r"], 0.952412402)
    assert np.isnan(at.loc['Colour', "Pearson's r"])
    assert np.isclose(at.loc['Apples', "Pearson's r"], 0.329895861)
