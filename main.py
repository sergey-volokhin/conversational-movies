import argparse
import json
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor as GBRT
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from surprise import SVD, Dataset, KNNBaseline, Reader, SVDpp
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_pandas

from model_utils import (bc, datapath, get_average_score, logger, movies_features, critics_reviews,
                         model_features, searcher, timeit, whoosh_parser)

seed = 42
np.random.seed(42)

num_bert_feats = 1024

def get_ranked_critics(movie_id, text):
    ''' returns dict of ranked according to text critics with BM25 weights '''
    query = whoosh_parser.parse(f'movie_id:{movie_id} AND (review:' + ' OR review:'.join(text.split()) + ')')
    initial_list = [(i['critic_id'], i.score) for i in searcher.search(query, limit=None)]
    if len(initial_list) > 0:
        biggest_weight = initial_list[0][1]
        return {i[0]: i[1] / biggest_weight for i in initial_list}
    logger.warning(f'found 0 bm25 reviews for {movie_id}')
    return {}


def get_movie_critic_representation(row):
    '''
        returns movie vector, created using embeddings
        of critics' reviews' and their BM25 weights
    '''
    critics_dict = get_ranked_critics(row.movie_id, row.review)
    subdf = critics_reviews[critics_reviews['movie_id'] == row.movie_id]
    subsubdf = subdf[subdf['critic_id'].isin(critics_dict.keys())]
    if len(critics_dict) > 0:
        encodings = bc.encode(subsubdf['review'].to_list(), show_progress_bar=False)
        weights = [critics_dict[name] for name in subsubdf['critic_id'].to_list()]
        weights = np.array(weights).reshape((len(weights), 1))
        return (encodings * weights).mean(axis=0)
    return np.zeros(num_bert_feats)


@timeit
def train_cf(df, model='knn'):
    reader = Reader(rating_scale=(1, 5))
    logger.info('training '+model)
    if model == 'knn':
        algo = KNNBaseline(sim_options={'name': 'cosine', 'user_based': True}, verbose=False)
    elif model == 'svd':
        algo = SVD(n_factors=500)
    elif model == 'svdpp':
        algo = SVDpp(n_factors=500)
    data_train = Dataset.load_from_df(df, reader)
    algo.fit(data_train.build_full_trainset())
    pickle.dump(algo, open(outpath + 'cf.pkl', 'wb'))


def convert_to_cf_matrix(df):
    ''' converts df_users into a standard records cf matrix '''
    result = pd.DataFrame()
    for number in ['first', 'second']:
        temp = df[['critic_id', number+' movie_id', number+' score']].drop_duplicates()
        temp = temp.rename(columns={number+' movie_id': 'movie_id', number+' score': 'score'})
        result = result.append(temp)
    return result


def get_cf_feature(df, regenerate):
    if not os.path.exists(outpath+'cf.pkl') or regenerate:
        df_users_cf = convert_to_cf_matrix(df)
        df_cf = critics_reviews[['critic_id', 'movie_id', 'score']].append(df_users_cf).reset_index(drop=True).dropna()
        train_cf(df_cf)
    cf = pickle.load(open(outpath+'cf.pkl', 'rb'))

    df['cf_score'] = df.apply(lambda x: cf.predict(x['critic_id'], x['target movie_id']).est, axis=1)
    logger.info(f"RMSE for CF: {math.sqrt(mse(df['cf_score'], df['target score']))}")
    logger.info(f" MAE for CF: {mae(df['cf_score'], df['target score'])}")
    df.to_csv(outpath+'conversations_w_cf.tsv', sep='\t', index=False)


@timeit
def get_users_emb(df, each_turn='full_text'):
    '''
        options for embedding calculation:
            each_turn: takes the average embedding of turns' embeddings
            full_text: takes the embedding of all text concatenated
    '''
    logger.info('getting user embedding')
    columns = [f'user_{i}' for i in range(num_bert_feats)]
    if each_turn == 'each_turn':
        embedded = [bc.encode(i, show_progress_bar=False).mean(axis=0) for i in df['review'].progress_apply(lambda x: x.split(' ||| ')).to_list()]
        user_emb = pd.DataFrame.from_records(embedded, columns=columns)
    else:
        user_emb = pd.DataFrame.from_records(bc.encode(df['review'].to_list(), show_progress_bar=False), columns=columns)
    user_emb.to_csv(outpath+f'users_emb_{each_turn}.tsv', sep='\t', index=False)


@timeit
def get_critics_emb(df):
    logger.info('getting critic embedding')
    tqdm_pandas.pandas(desc='critics_emb')
    df_tmp = df.rename(columns={'target movie_id': 'movie_id'}).reset_index(drop=True)
    df_tmp = df_tmp.progress_apply(get_movie_critic_representation, axis=1)
    critics_emb = pd.DataFrame.from_records(df_tmp, columns=[f'critics_emb_{i}' for i in range(num_bert_feats)])
    critics_emb.to_csv(outpath+f'critics_emb.tsv', sep='\t', index=False)


@timeit
def get_metadata_features(df, user_vectors):
    logger.info('getting metadata features')
    new_feats = []
    for i, user in enumerate(tqdm(user_vectors, desc='metadata features')):
        row = df.iloc[i].copy()
        current_movie_features = movies_features[row['target movie_id']]
        for feature in ['genre', 'people', 'description', 'title']:
            if feature not in current_movie_features or current_movie_features[feature] == []:
                logger.warning(f"no {feature} for {row['target movie_id']}")
        if 'audience_score' not in current_movie_features or 'critic_score' not in current_movie_features:
            logger.warning('no scores for ', row['target movie_id'])
            row['meta_audience_score'] = np.nan
            row['meta_critics_score'] = np.nan
        else:
            row['meta_audience_score'] = current_movie_features['audience_score'] / 100
            row['meta_critics_score'] = current_movie_features['critic_score'] / 100

        to_encode = [', '.join(current_movie_features['genre']),
                     ', '.join(current_movie_features['people'][:10]),
                     current_movie_features['description'],
                     current_movie_features['title']]
        for i in range(len(to_encode)):
            if len(to_encode[i]) == 0:
                to_encode[i] = 'None'
        genre_vector, people_vector, descr_vector, title_vector = bc.encode(to_encode, show_progress_bar=False)

        row['meta_date'] = current_movie_features['in theaters']
        row['meta_runtime'] = current_movie_features['runtime']
        row['meta_amount_critics'] = current_movie_features['amount of critics']
        row['meta_amount_users'] = current_movie_features['amount of users']
        row['meta_genre'] = np.dot(user, np.transpose(genre_vector))
        row['meta_people'] = np.dot(user, np.transpose(people_vector))
        row['meta_description'] = np.dot(user, np.transpose(descr_vector))
        row['meta_title'] = np.dot(user, np.transpose(title_vector))
        new_feats.append(row)
    return pd.DataFrame.from_records(new_feats)


def get_all_features(df_users, regenerate=False):
    """ prepare a dataframe with all the features """

    logger.info('generating data')
    if not os.path.exists(outpath+'conversations_w_cf.tsv') or regenerate:
        get_cf_feature(df_users, regenerate)
    if not os.path.exists(outpath+f'users_emb_full_text.tsv') or regenerate:
        get_users_emb(df_users)
    if not os.path.exists(outpath+f'critics_emb.tsv') or regenerate:
        get_critics_emb(df_users)

    df_users = pd.read_table(outpath+'conversations_w_cf.tsv')
    user_vectors = pd.read_table(outpath+f'users_emb_full_text.tsv').values.tolist()
    critics_vectors = pd.read_table(outpath+f'critics_emb.tsv').values.tolist()

    earth_movers_distance = [[stats.wasserstein_distance(critics_vectors[i], user)] for i, user in enumerate(user_vectors)]
    dot_product = [[np.dot(user, np.transpose(critics_vectors[i]))] for i, user in enumerate(user_vectors)]
    emd_df = pd.DataFrame.from_records(earth_movers_distance, columns=['difference'])
    dp_df = pd.DataFrame.from_records(dot_product, columns=['dot_product'])

    df_users = pd.concat([df_users.reset_index(drop=True), emd_df, dp_df], axis=1)
    df_all_features = get_metadata_features(df_users, user_vectors)
    df_all_features.to_csv(df_all_features_path, sep='\t', index=False)


def avg_critics(row):
    return movies_features[row['target movie_id']]['critic_score'] / 20


def avg_audience(row):
    return movies_features[row['target movie_id']]['audience_score'] / 20


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--regenerate',
                        help='force regenerate all data, embeddings and model',
                        action='store_true')
    parser.add_argument('-d', '--directory',
                        help='directory where everything is saved',
                        default='tmp')
    parser.add_argument('-f', '--feature-importances',
                        help='whether to print feature importances of the model',
                        action='store_true')
    args = parser.parse_args()
    outpath = f'{datapath}/../{args.directory}/'

    df_users = pd.read_table(outpath+f'conversations_estimated.tsv')
    df_users = df_users[df_users['target movie_id'].isin(movies_features)]

    df_all_features_path = outpath+f'conversations_w_features.tsv'
    if not os.path.exists(df_all_features_path) or args.regenerate:
        get_all_features(df_users, args.regenerate)
    df_all_features = pd.read_table(df_all_features_path)

    model = GBRT(n_estimators=10, max_depth=3)
    X, y = df_all_features[model_features+['target movie_id']], df_all_features['target score']
    get_average_score(X.drop('target movie_id', axis=1), y, model, n=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    model.fit(X_train.drop('target movie_id', axis=1), y_train)
    y_pred = model.predict(X_test.drop('target movie_id', axis=1))
    logger.info(f'RMSE: {math.sqrt(mse(y_test, y_pred))}')
    logger.info(f' MAE: {mae(y_test, y_pred)}')
    pickle.dump(model, open(outpath + f'gbdt.pkl', 'wb'))

    if args.feature_importances:
        logger.info('Feature importances:')
        for score, feature in sorted(zip(model.feature_importances_, model_features), reverse=True):
            logger.info(f'{round(score, 4)} {feature}')

    # _stat, p = stats.wilcoxon(X_test['cf_score'], X_test.apply(avg_audience, axis=1))
    # logger.info(f'wilcoxon test p-value: avg_audience & gbdt: {p}')
    # _stat, p = stats.wilcoxon(y_pred, X_test['cf_score'])
    # logger.info(f'wilcoxon test p-value: cf & gbdt: {p}')

    # rmses = {}
    # ''' ablation '''
    # for columns, name in zip([[], ['cf_score'], ['difference'], ['dot_product'], ['meta_audience_score', 'meta_critics_score', 'meta_date', 'meta_runtime', 'meta_amount_critics', 'meta_amount_users', 'meta_genre', 'meta_people', 'meta_description', 'meta_title']], ['full model', 'cf', 'diff', 'dot_product', 'meta']):

    #     logger.info(name)
    #     df_all_features = pd.read_table(df_all_features_path)

    #     X, y = df_all_features[[i for i in model_features if i not in columns]+['target movie_id']], df_all_features['target score']
    #     model = GBRT(n_estimators=10, max_depth=3)
    #     rmses[name] = get_average_score(X.drop('target movie_id', axis=1), y, model, n=100)

    # plt.boxplot(rmses.values(), labels=rmses.keys())
    # plt.savefig(outpath+'ablation.png')
