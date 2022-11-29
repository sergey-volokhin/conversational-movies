import math
import os
import pickle

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as GBRT
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split as tts
from tqdm import trange


class MyModel:

    def __init__(self, args):
        self.logger = args.logger
        self.seed = args.seed
        self.outpath = args.outpath
        self.feature_importances = args.feature_importances
        self.model = GBRT(n_estimators=10, max_depth=3)

    def get_average_score(self, x, y, n=100):
        rmses = []
        maes = []
        for _ in trange(n, leave=False, desc='running models'):
            x_train, x_test, y_train, y_test = tts(x, y, test_size=0.1)
            self.model.fit(x_train, y_train)
            predictions = self.model.predict(x_test)
            rmses.append(math.sqrt(mse(y_test, predictions)))
            maes.append(mae(y_test, predictions))
        rmses = np.array(rmses)
        maes = np.array(maes)
        self.logger.info(f'getting average metrics on {n} {type(self.model).__name__}s')
        self.logger.info(f'avg RMSE: {rmses.mean():.4f}, std: {np.std(rmses):.4f}')
        self.logger.info(f'avg MAE:  {maes.mean():.4f}, std: {np.std(maes):.4f}')
        return rmses

    def fit(self, dataset):
        X, y = dataset.data[dataset.features + ['movie_3']], dataset.data['score_3']
        self.get_average_score(X.drop('movie_3', axis=1), y)
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
