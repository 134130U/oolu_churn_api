import pandas as pd
import numpy as np
from Scripts import config
from datetime import date

now = date.today()  # '2020-10-16'


class Preprocess(object):
    def __init__(self, select_features=config.FEATURES, today=now):
        self.select_features = select_features
        self.today = pd.to_datetime(str(today), errors='ignore')

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "Preprocess":
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame):
        X = X[self.select_features]
        X['prev_payment'] = X['prev_payment'].fillna(X['created_at'])
        X['created_at'] = pd.to_datetime(X['created_at'], errors='ignore')
        X['prev_payment'] = pd.to_datetime(X['prev_payment'], errors='ignore')

        # Let's set up the date we got the inputs (27/08/2019) to compute the Age of the customer account in month
        X['AccountAge'] = (self.today - X['created_at']) / np.timedelta64(1, 'M')
        # Let's compute the number of days since the previous payment
        X['SinceLastPay'] = (self.today - X['prev_payment']) / np.timedelta64(1, 'D')
        X.drop(['created_at', 'prev_payment'], axis=1, inplace=True)

        return X