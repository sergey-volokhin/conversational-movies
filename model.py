import math
import os
import pickle

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as GBRT
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_validate
from tqdm import trange


class MyModel:

    def __init__(self, args):
        self.logger = args.logger
        self.seed = args.seed
        self.outpath = args.outpath
        self.feature_importances = args.feature_importances
        self.model = GBRT(n_estimators=10, max_depth=3)

    def fit(self, dataset):
        X, y = dataset.data[dataset.features + ['movie_3']], dataset.data['score_3']

        cv_result = cross_validate(
            self.model,
            X.drop('movie_3', axis=1),
            y,
            cv=5,
            scoring=['neg_mean_squared_error', 'neg_mean_absolute_error'],
        )

        for key, value in cv_result.items():
            if key.startswith('test'):
                self.logger.info(f'{key}: {abs(value.mean()):.4f} ± {value.std():.4f}')

        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=self.seed)
        self.model.fit(X_train.drop('movie_3', axis=1), y_train)
        y_pred = self.model.predict(X_test.drop('movie_3', axis=1))
        self.logger.info(f'RMSE: {math.sqrt(mse(y_test, y_pred)):.4f}')
        self.logger.info(f' MAE: {mae(y_test, y_pred):.4f}')
        pickle.dump(self.model, open(os.path.join(self.outpath, 'gbdt.pkl'), 'wb'))

        if self.feature_importances:
            self.logger.info('Feature importances:')
            for score, feature in sorted(zip(self.model.feature_importances_, dataset.features), reverse=True):
                self.logger.info(f'{score:.4f} {feature}')
