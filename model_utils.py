import json
import logging
import math
import os
import sys
import time
import pandas as pd

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split as tts
from tqdm import trange
from whoosh import index, scoring, fields, qparser

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()

datapath = os.path.dirname(os.path.abspath(sys.argv[0])) + '/data/'
conversations = json.load(open(datapath + 'MovieSent.json', 'r'))
movie_ids = json.load(open(datapath + 'films_rt_ids.json', 'r'))
movies_features = json.load(open(datapath + 'films_features.json', 'r'))
try:
    critics_reviews = pd.read_table(datapath+'reviews.tsv').drop(['fresh'], axis=1).dropna()
except FileNotFoundError:
    logger.error('Critics reviews not found. Please, run `scraping_critics.py`')
    exit()

bc = SentenceTransformer('stsb-roberta-large')

schema = fields.Schema(movie_id=fields.KEYWORD(stored=True, scorable=True),
                critic_id=fields.KEYWORD(stored=True, scorable=True),
                score=fields.NUMERIC(stored=True),
                review=fields.TEXT(stored=True),
                freshness=fields.KEYWORD(scorable=True))
whoosh_parser = qparser.MultifieldParser(['movie_id', 'review'], schema=schema)
searcher = index.open_dir(datapath + 'index').searcher(weighting=scoring.BM25F)

model_features = ['cf_score',
                  'difference',
                  'dot_product',
                  'meta_genre',
                  'meta_people',
                  'meta_description',
                  'meta_title',
                  'meta_runtime',
                  'meta_date',
                  'meta_audience_score',
                  'meta_critics_score',
                  'meta_amount_critics',
                  'meta_amount_users']

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f'{func.__name__} took {round(time.time()-start)} seconds to complete')
        return result
    return wrapper


def get_average_score(X, y, model, n=100):
    rmses = []
    maes = []
    for _ in trange(n, leave=False, desc='running models'):
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmses.append(math.sqrt(mse(y_test, predictions)))
        maes.append(mae(y_test, predictions))
    rmses = np.array(rmses)
    maes = np.array(maes)
    logger.info(f'getting average metrics on {n} {type(model).__name__}s')
    logger.info(f'avg RMSE: {round(rmses.mean(), 4)}, std: {round(np.std(rmses), 3)}')
    logger.info(f'avg MAE:  {round(maes.mean(), 4)}, std: {round(np.std(maes), 3)}')
    return rmses
