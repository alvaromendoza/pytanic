"""Cross-validate machine learning models."""

# Fix random seeds for reproducibility

from titanic.config import RANDOM_SEED
from numpy.random import seed
seed(RANDOM_SEED)
import os
os.environ['PYTHONHASHSEED'] = '0'
import random as rn
rn.seed(RANDOM_SEED)

# Main imports

import time
import pprint
import numpy as np
import titanic.tools as tools
from titanic.modelling import SimpleDataFrameImputer, DataFrameDummifier, CategoricalToString
from titanic.modelling import ExtendedClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from category_encoders.ordinal import OrdinalEncoder


def cross_validate_logreg(X_train, y_train, pipes, grids, kfolds):
    pipes['logreg'] = make_pipeline(SimpleDataFrameImputer(median_cols=['Age', 'Fare'],
                                                           mode_cols=['Embarked']),
                                    DataFrameDummifier(),
                                    LogisticRegression(solver='liblinear'))
    grids['logreg'] = {'logisticregression__C': [0.01, 0.1, 0.5, 0.8, 1, 1.2, 2, 5, 10]}
    grids['logreg'] = {'logisticregression__C': [0.6, 0.75, 0.8, 0.85, 0.9]}

    logreg = ExtendedClassifier.cross_validate(pipes['logreg'], X_train, y_train, grids['logreg'],
                                               sklearn_gscv_kws={'cv': 3},
                                               sklearn_cvs_kws={'cv': kfolds},
                                               param_strategy='best',
                                               logdir_path=r'logs/models/logreg',
                                               serialize_to=r'models/logreg.pickle')
    return logreg


def cross_validate_forest(X_train, y_train, pipes, grids, kfolds, random_search=False):
    pipes['forest'] = make_pipeline(CategoricalToString(),
                                    SimpleDataFrameImputer(median_cols=['Age', 'Fare'],
                                                           mode_cols=['Embarked']),
                                    OrdinalEncoder(cols=['Title', 'Deck', 'Embarked'],
                                                   handle_unknown='impute'),
                                    RandomForestClassifier(**{'bootstrap': True,
                                                              'max_depth': 70,
                                                              'max_features': 'auto',
                                                              'min_samples_leaf': 4,
                                                              'min_samples_split': 10,
                                                              'n_estimators': 64,
                                                              'random_state': RANDOM_SEED}))
    if random_search:
        n_estimators = [int(x) for x in np.linspace(start=10, stop=500, num=10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]

        random_grid = {'randomforestclassifier__n_estimators': n_estimators,
                       'randomforestclassifier__max_features': max_features,
                       'randomforestclassifier__max_depth': max_depth,
                       'randomforestclassifier__min_samples_split': min_samples_split,
                       'randomforestclassifier__min_samples_leaf': min_samples_leaf,
                       'randomforestclassifier__bootstrap': bootstrap}
        pprint.pprint(random_grid)
        randsearch = RandomizedSearchCV(pipes['forest'], random_grid, n_iter=50, cv=3,
                                        verbose=0, random_state=42)
        start = time.time()
        randsearch.fit(X_train, y_train)
        finish = time.time()
        print('randsearch.fit execution time:', finish - start)
        pprint.pprint(randsearch.best_params_)

    forest = ExtendedClassifier.cross_validate(pipes['forest'], X_train, y_train,
                                               sklearn_cvs_kws={'cv': kfolds},
                                               param_strategy='init',
                                               logdir_path=r'logs/models/forest',
                                               serialize_to=r'models/forest.pickle')
    return forest


def cross_validate_svc(X_train, y_train, pipes, grids, kfolds):
    pipes['svc'] = make_pipeline(SimpleDataFrameImputer(median_cols=['Age', 'Fare'],
                                                        mode_cols=['Embarked']),
                                 DataFrameDummifier(),
                                 SVC(kernel='linear', C=0.1))
    C = [0.001, 0.01, 0.1, 1, 10]
    gamma = [0.001, 0.01, 0.1, 1]

    grids['svc'] = {'svc__C': C, 'svc__gamma': gamma}

    svc = ExtendedClassifier.cross_validate(pipes['svc'], X_train, y_train,
                                            sklearn_cvs_kws={'cv': kfolds},
                                            param_strategy='init',
                                            logdir_path=r'logs/models/svc',
                                            serialize_to=r'models/svc.pickle')
    return svc


def cross_validate_models():
    # Load data

    X_train = tools.deserialize(r'data/processed/X_train.pickle')
    y_train = tools.deserialize(r'data/processed/y_train.pickle')

    # Create global objects

    pipes = dict()
    grids = dict()
    kfolds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Train models

    cross_validate_logreg(X_train, y_train, pipes, grids, kfolds)
    cross_validate_forest(X_train, y_train, pipes, grids, kfolds)
    cross_validate_svc(X_train, y_train, pipes, grids, kfolds)


if __name__ == '__main__':
    cross_validate_models()
