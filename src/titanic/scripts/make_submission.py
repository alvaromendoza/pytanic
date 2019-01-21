import pandas as pd
import titanic.tools as tools
from titanic.modelling import ExtendedClassifier


def make_submission():
    X_train = tools.deserialize(r'data/processed/X_train.pickle')
    X_test = tools.deserialize(r'data/processed/X_test.pickle')
    y_train = tools.deserialize(r'data/processed/y_train.pickle')
    model = ExtendedClassifier.deserialize(r'models/voting.pickle')
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    submission = pd.read_csv(r'data/raw/gender_submission.csv')
    submission['Survived'] = prediction
    submission.to_csv('results/submission.csv', index=False)


if __name__ == '__main__':
    make_submission()
