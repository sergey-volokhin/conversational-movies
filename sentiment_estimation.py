import argparse
import math
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split as tts

from model_utils import (bc, conversations, datapath, logger, movie_ids,
                         searcher, timeit, whoosh_parser)

np.random.seed(42)


def entities_in_utterance(utterance):
    """ extract entities from the utterance """
    if 'segments' not in utterance:
        return set()
    ents = set()
    for segment in utterance['segments']:
        entity_type = 'entityType' in segment['annotations'][0] and segment['annotations'][0]['entityType'] == 'MOVIE_OR_SERIES'
        annotation_type = 'annotationType' in segment['annotations'][0] and segment['annotations'][0]['annotationType'] == 'ENTITY_NAME'
        if entity_type and annotation_type:
            if 'entityId' in segment['annotations'][0]:
                ents.add(segment['annotations'][0]['entityId'])
            else:
                ents.add(segment['text'])
    return ents


def merge_utts(utterances):
    """ merge utterances from the same speaker that apper in a row """
    cur_agent = utterances[0]['speaker']
    result = ''
    for utterance in utterances:
        if utterance['speaker'] == cur_agent:
            result += ' ' + utterance['text']
        else:
            result += ' ||| ' + utterance['text']
            cur_agent = utterance['speaker']
    return result.strip()


def get_sentiment_from_utterance(utterance):
    """ get the judge sentiment for the first entity in the utterance """
    for segment in utterance['segments']:
        if 'annotations' in segment and 'sentiment' in segment['annotations'][0]:
            intial_sentiment = segment['annotations'][0]['sentiment']
            if np.isnan(intial_sentiment[0]):
                return intial_sentiment[-1]
            elif np.isnan(intial_sentiment[-1]):
                return intial_sentiment[0]
            return sum(intial_sentiment) / len(intial_sentiment)
    logger.error('sentiment not extracted from:', utterance)


def sentiment_2_score(x):
    """ convert [-3:+3] score to a [1:5] rating """
    try:
        return round((2 * x + 9) / 3)
    except Exception:
        return np.nan


def get_gbdt_conversations(conversations):
    """ convert original json conversations into pandas df
        and add estimated score
    """
    zero_reviewed = []
    for movie_id in movie_ids:
        results = searcher.search(whoosh_parser.parse(f'movie_id:{movie_id}'), limit=None)
        if len(results) == 0:
            zero_reviewed.append(movie_id)

    list_of_lines = []
    for conversation in conversations:
        entities = set()
        line = {'critic_id': conversation['conversationId']}
        for utterance in conversation['utterances']:
            cur_ents = entities_in_utterance(utterance)
            if utterance['speaker'] == 'USER' and 'segments' in utterance and len(cur_ents) == 1:
                movie_id = list(cur_ents)[0]
                movie_sentiment = sentiment_2_score(get_sentiment_from_utterance(utterance))
                valid_id = movie_id not in entities and movie_id not in zero_reviewed and movie_id in movie_ids
                valid_score = len(entities) < 2 or not np.isnan(movie_sentiment)
                if valid_id and valid_score:
                    if len(entities) == 0:
                        line['first movie_id'] = movie_id
                        line['first score'] = movie_sentiment
                        line['first text'] = get_context(conversation, utterance)
                    elif len(entities) == 1:
                        line['second movie_id'] = movie_id
                        line['second score'] = movie_sentiment
                        line['second text'] = get_context(conversation, utterance)
                        line['review'] = merge_utts(conversation['utterances'][:utterance['index'] + 1])
                    elif len(entities) == 2:
                        line['target movie_id'] = movie_id
                        line['target score'] = movie_sentiment
                        list_of_lines.append(line)
                        break
                    entities.add(movie_id)
    return pd.DataFrame.from_records(list_of_lines)


def get_context(conversation, utterance):
    if utterance['index'] > 0 and conversation['utterances'][utterance['index'] - 1]['speaker'] == 'ASSISTANT':
        text = conversation['utterances'][utterance['index'] - 1]['text'] + ' ||| ' + utterance['text']
    else:
        text = utterance['text']
    return text


def create_sentiment_trainset(file):
    """ use utterances from conversations that can't be used
        for the final model to train sentiment estimator on them
    """
    forbidden_pairs = []
    for column in ['first movie_id', 'second movie_id', 'target movie_id']:
        forbidden_pairs += [list(i) for i in df_conv[['critic_id', column]].values]

    list_of_lines = []
    for conversation in conversations:
        for utterance in conversation['utterances']:
            line = {'critic_id': conversation['conversationId']}
            cur_ents = entities_in_utterance(utterance)
            if utterance['speaker'] == 'USER' and 'segments' in utterance and len(cur_ents) == 1:
                movie_id = list(cur_ents)[0]
                movie_sentiment = sentiment_2_score(get_sentiment_from_utterance(utterance))
                if [conversation['conversationId'], movie_id] not in forbidden_pairs and not np.isnan(movie_sentiment):
                    line['movie_id'] = movie_id
                    line['score'] = movie_sentiment
                    line['text'] = get_context(conversation, utterance)
                    list_of_lines.append(line)
    trainset = pd.DataFrame.from_records(list_of_lines)
    trainset[['text', 'score']].to_csv(file, sep='\t', index=False)


@timeit
def train_estimator(sentiment_df):

    # embed the train dataset
    emb_path = outpath + 'trainset_embeddings.txt'
    if args.regenerate or not os.path.exists(model_path):
        logger.info('Calculating embeddings')
        sentiment_estimation_embeddings = bc.encode(sentiment_df['text'].to_list())
        logger.info('Embeddings calculated')
        np.savetxt(emb_path, sentiment_estimation_embeddings)
    sentiment_estimation_embeddings = np.loadtxt(emb_path)

    # remove those few conversations which have score 3 to reduce noise
    indexes = sentiment_df[sentiment_df['score'] == 3].index.to_list()
    sentiment_df = sentiment_df[sentiment_df['score'] != 3]
    sentiment_estimation_embeddings = [i for ind, i in enumerate(sentiment_estimation_embeddings) if ind not in indexes]

    logger.info('Fitting the estimator')
    x_train, x_test, y_train, y_test = tts(sentiment_estimation_embeddings, sentiment_df['score'], test_size=0.1)
    model = RFR(n_estimators=500, max_depth=10, n_jobs=-1)
    model.fit(x_train, y_train)
    logger.info('Estimator fitted')

    pickle.dump(model, open(model_path, 'wb'))
    y_pred = model.predict(x_test)
    logger.info(f'sentiment estimator RMSE: {round(math.sqrt(mse(y_test, y_pred)), 3)}')
    logger.info(f'sentiment estimator  MAE: {round(mae(y_test, y_pred), 3)}')
    logger.info(f'sentiment estimator  R^2: {round(model.score(x_test, y_test), 3)}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--regenerate',
                        help='force regenerate all data, embeddings and model',
                        action='store_true')
    parser.add_argument('-d', '--directory',
                        help='directory where the model is saved',
                        default='tmp/sentiment',
                        action='store_true')
    args = parser.parse_args()

    df_conv = get_gbdt_conversations(conversations)
    outpath = f'{datapath}/../{args.directory}/'
    model_path = outpath + 'estimator.pickle'
    trainset_path = outpath + 'trainset.tsv'

    os.makedirs(outpath, exist_ok=True)
    if args.regenerate or not os.path.exists(model_path):
        if args.regenerate or not os.path.exists(trainset_path):
            create_sentiment_trainset(trainset_path)
        sentiment_df = pd.read_table(trainset_path)
        train_estimator(sentiment_df)
    model = pickle.load(open(model_path, 'rb'))

    logger.info('Encoding conversations')
    embeddings = bc.encode(df_conv['first text'].tolist() + df_conv['second text'].tolist(), show_progress_bar=False)
    df_conv['first score'] = model.predict(embeddings[:df_conv.shape[0]])
    df_conv['second score'] = model.predict(embeddings[df_conv.shape[0]:])
    logger.info('Encoding done')

    df_conv.drop(['first text', 'second text'], axis=1, inplace=True)
    df_conv.to_csv(outpath + '../conversations_estimated.tsv', sep='\t', index=False)
