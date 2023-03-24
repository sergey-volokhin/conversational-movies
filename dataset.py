import json
import math
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split as tts
from surprise import SVD, Dataset, KNNBaseline, Reader, SVDpp
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_pandas

from indexing import load_index


def entities_in_utterance(utterance):
    """ extract entities from the utterance """
    if 'segments' not in utterance:
        return set()
    ents = set()
    for segment in utterance['segments']:
        annotation = segment['annotations'][0]
        entity_type = 'entityType' in annotation and annotation['entityType'] == 'MOVIE_OR_SERIES'
        annotation_type = 'annotationType' in annotation and annotation['annotationType'] == 'ENTITY_NAME'
        if entity_type and annotation_type:
            if 'entityId' in annotation:
                ents.add(annotation['entityId'])
            else:
                ents.add(segment['text'])
    return list(ents)


def get_context(dialog, utterance, sep='|||'):
    ''' append previous turn to current utterance to create context '''
    prev_utterance = dialog['utterances'][utterance['index'] - 1]
    if utterance['index'] > 0 and prev_utterance['speaker'] == 'ASSISTANT':
        return prev_utterance['text'] + f' {sep} ' + utterance['text']
    return utterance['text']


def sentiment_to_score(x):
    """ convert [-3:+3] score to a [1:5] rating """
    try:
        return round((2 * x + 9) / 3)
    except Exception:
        return np.nan


def merge_utts(utterances, sep='|||'):
    """ merge utterances from the same speaker that apper in a row """
    cur_agent = utterances[0]['speaker']
    result = ''
    for utterance in utterances:
        if utterance['speaker'] == cur_agent:
            result += ' ' + utterance['text']
        else:
            result += f' {sep} ' + utterance['text']
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


class MyDataset:

    features = [
        'cf_score',
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
        'meta_amount_users'
    ]

    def __init__(self, args):

        self.logger = args.logger
        self.outpath = args.outpath
        self.datapath = os.path.join(os.path.dirname(self.outpath), 'data')
        self.regenerate = args.regenerate
        self.cf_type = args.cf_type
        self.seed = args.seed
        self.sep = args.sep
        self.num_bert_feats = 768
        self.bc = SentenceTransformer(args.bert_model)

        self.load_data()
        self.parser, self.searcher = load_index(self.datapath, self.reviews)
        self.process_dialogs()

    def load_data(self):
        self.reviews = pd.read_table(os.path.join(self.datapath, 'reviews.tsv')).drop(['fresh'], axis=1).dropna()
        self.movie_ids = json.load(open(os.path.join(self.datapath, 'films_rt_ids.json'), 'r'))
        self.movies_features = json.load(open(os.path.join(self.datapath, 'films_features.json'), 'r'))
        self.raw_dialogs = json.load(open(os.path.join(self.datapath, 'MovieSent.json'), 'r'))

    def process_dialogs(self) -> None:
        """
        convert original json conversations into pandas df
        each row has:
            both movies idss
            both estimated score
            utterances about both movies
        """

        self.zero_reviewed_movies = []
        for movie_id in self.movie_ids:
            if not self.searcher.document(movie_id=movie_id):
                self.zero_reviewed_movies.append(movie_id)
        list_of_lines = [self.process_single_dialog(dialog) for dialog in self.raw_dialogs]
        self.dialogs = pd.DataFrame.from_records([i for i in list_of_lines if i is not None])
        self.dialogs = self.dialogs[self.dialogs['movie_3'].isin(self.movies_features)]

    def process_single_dialog(self, dialog):
        '''
        for each user utterance with exactly one entity (movie):
            extract movie_id, golden score, text with context
        return the first 3 results (2 train movie, 1 test movie)
        '''

        entities = []
        line = {'critic_id': dialog['conversationId']}
        for utterance in dialog['utterances']:

            cur_ents = entities_in_utterance(utterance)
            if utterance['speaker'] != 'USER' or 'segments' not in utterance or len(cur_ents) != 1:
                continue

            movie_id = cur_ents[0]
            movie_sentiment = sentiment_to_score(get_sentiment_from_utterance(utterance))
            valid_id = movie_id not in entities + self.zero_reviewed_movies and movie_id in self.movie_ids
            valid_score = len(entities) < 2 or not np.isnan(movie_sentiment)

            if valid_id and valid_score:
                line[f'movie_{len(entities)+1}'] = movie_id
                line[f'score_{len(entities)+1}'] = movie_sentiment
                if len(entities) != 2:
                    line[f'text_{len(entities)+1}'] = get_context(dialog, utterance, sep=self.sep)

                if len(entities) == 1:
                    line['review'] = merge_utts(dialog['utterances'][:utterance['index'] + 1], sep=self.sep)
                elif len(entities) == 2:
                    return line

                entities.append(movie_id)

    def load_all_features(self) -> pd.DataFrame:
        path = self.get_path('all_features.tsv')
        if not self.regenerate and os.path.exists(path):
            self.data = pd.read_table(path)

        os.makedirs(self.outpath, exist_ok=True)
        df_cf = self.get_cf_feature()
        df_adaptive = self.get_adaptive_features(df_cf)
        self.data = self.get_metadata_features(df_adaptive)
        self.data.to_csv(path, sep='\t', index=False)

    def get_users_emb(self, df, each_turn='full_text', sep='|||') -> pd.DataFrame:
        '''
            embed the user conversations
            optional:
                each_turn: takes the average embedding of turns' embeddings
                full_text: takes the embedding of all text concatenated
        '''
        path = self.get_path(f'users_emb_{each_turn}.tsv')
        if not self.regenerate and os.path.exists(path):
            return pd.read_table(path).values.tolist()

        self.logger.info('getting user embedding')
        columns = [f'user_{i}' for i in range(self.num_bert_feats)]
        if each_turn == 'each_turn':
            to_embed = df['review'].progress_apply(lambda x: x.split(f' {sep} ')).to_list()
            embedded = [self.bc.encode(i, show_progress_bar=False, batch_size=1024).mean(axis=0) for i in to_embed]
            user_emb = pd.DataFrame.from_records(embedded, columns=columns)
        else:
            user_emb = pd.DataFrame.from_records(
                self.bc.encode(
                    df['review'].to_list(),
                    show_progress_bar=False,
                    batch_size=1024
                ),
                columns=columns
            )
        user_emb.to_csv(path, sep='\t', index=False)
        return user_emb.values.tolist()

    def get_critics_emb(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' represent critics are the average of their reviews '''
        path = self.get_path('critics_emb.tsv')
        if not self.regenerate and os.path.exists(path):
            return pd.read_table(path).values.tolist()

        self.logger.info('getting critic embedding')
        tqdm_pandas.pandas(desc='critics_emb', dynamic_ncols=True)
        df_tmp = df.rename(columns={'movie_3': 'movie_id'}).reset_index(drop=True)
        critics_emb = pd.DataFrame.from_records(
            df_tmp.progress_apply(self.get_movie_critic_representation, axis=1),
            columns=[f'critics_emb_{i}' for i in range(self.num_bert_feats)]
        )
        critics_emb.to_csv(path, sep='\t', index=False)
        return critics_emb.values.tolist()

    def get_movie_critic_representation(self, row):
        '''
        returns movie representation:
            weighted (by BM25 scores) average of critics' reviews for that movie
        '''
        critics_dict = self.get_ranked_critics(row.movie_id, row.review)
        subdf = self.reviews[self.reviews['movie_id'] == row.movie_id]
        subsubdf = subdf[subdf['critic_id'].isin(critics_dict.keys())]
        if len(critics_dict) > 0:
            encodings = self.bc.encode(subsubdf['review'].to_list(), show_progress_bar=False, batch_size=1024)
            weights = [critics_dict[name] for name in subsubdf['critic_id'].to_list()]
            weights = np.array(weights).reshape((len(weights), 1))
            return (encodings * weights).mean(axis=0)  # average of reviews for one critic
        return np.zeros(self.num_bert_feats)

    def get_ranked_critics(self, movie_id, text):
        '''
        returns critics most similar to users according to BM25:
            {critic_id: BM25_score, ...}
        '''
        query = self.parser.parse(f'movie_id:{movie_id} AND (review:' + ' OR review:'.join(text.split()) + ')')
        initial_list = [(i['critic_id'], i.score) for i in self.searcher.search(query, limit=None)]
        if len(initial_list) > 0:
            biggest_weight = initial_list[0][1]
            return {i[0]: i[1] / biggest_weight for i in initial_list}
        self.logger.warning(f'found 0 bm25 reviews for {movie_id}')
        return {}

    def get_adaptive_features(self, df) -> pd.DataFrame:
        '''
        calculate features related to the distance between user and critics:
       `     retrieve the most similar to user critics with scores
            represent user as a weighted average of critics embeddings (weights are BM25 scores)
            calculate different metrics between that user representation and the conversation embeddings
        '''

        self.u_vectors = self.get_users_emb(df, sep=self.sep)
        self.c_vectors = self.get_critics_emb(df)

        path = self.get_path('adaptive_features.tsv')
        if not self.regenerate and os.path.exists(path):
            return pd.read_table(path)

        earth_movers_df = []
        dot_prod_df = []
        for i, user in enumerate(self.u_vectors):
            earth_movers_df.append([wasserstein_distance(self.c_vectors[i], user)])
            dot_prod_df.append([np.dot(user, np.transpose(self.c_vectors[i]))])
        earth_movers_df = pd.DataFrame.from_records(earth_movers_df, columns=['difference'])
        dot_prod_df = pd.DataFrame.from_records(dot_prod_df, columns=['dot_product'])

        features_df = pd.concat([df.reset_index(drop=True), earth_movers_df, dot_prod_df], axis=1)
        features_df.to_csv(path, sep='\t', index=False)
        return features_df

    def embed_text(self, sentences, file):
        if not self.regenerate and os.path.exists(file):
            return np.loadtxt(file)

        embeddings = self.bc.encode(sentences, show_progress_bar=False, batch_size=1024)
        np.savetxt(file, embeddings)
        return embeddings

    def get_path(self, file) -> str:
        return os.path.join(self.outpath, file)

    ''' CF stuff '''

    def train_cf(self) -> KNNBaseline | SVD | SVDpp:

        cf_path = self.get_path(f'cf_{self.cf_type}.pkl')
        if not self.regenerate and os.path.exists(cf_path):
            return pickle.load(open(cf_path, 'rb'))

        reader = Reader(rating_scale=(1, 5))
        self.logger.info(f'training {self.cf_type}')
        if self.cf_type == 'knn':
            cf_model = KNNBaseline(sim_options={'name': 'cosine', 'user_based': True}, verbose=False)
        elif self.cf_type == 'svd':
            cf_model = SVD(n_factors=500)
        elif self.cf_type == 'svdpp':
            cf_model = SVDpp(n_factors=500)

        data = self.convert_to_cf_matrix()
        data_train = Dataset.load_from_df(data, reader)
        cf_model.fit(data_train.build_full_trainset())
        pickle.dump(cf_model, open(cf_path, 'wb'))
        return cf_model

    def convert_to_cf_matrix(self):
        ''' converts df_users into a standard records cf matrix '''
        result = []
        for i in range(1, 3):
            temp = self.dialogs[['critic_id', f'movie_{i}', f'score_{i}']].drop_duplicates()
            temp = temp.rename(columns={f'movie_{i}': 'movie_id', f'score_{i}': 'score'})
            result.append(temp)
        return pd.concat([self.reviews[['critic_id', 'movie_id', 'score']], *result]).reset_index(drop=True).dropna()

    def create_sentiment_trainset(self):
        """
        use utterances from conversations that can't be used
        for the final model to train sentiment estimator on them
        """
        path = self.get_path('trainset.tsv')
        if not self.regenerate and os.path.exists(path):
            return pd.read_table(path)

        forbidden_pairs = []
        for i in range(1, 4):
            forbidden_pairs += [list(i) for i in self.dialogs[['critic_id', f'movie_{i}']].values]

        list_of_lines = []
        for dialog in self.raw_dialogs:
            for utterance in dialog['utterances']:
                line = {'critic_id': dialog['conversationId']}
                cur_ents = entities_in_utterance(utterance)
                if utterance['speaker'] == 'USER' and 'segments' in utterance and len(cur_ents) == 1:
                    movie_id = cur_ents[0]
                    movie_sentiment = sentiment_to_score(get_sentiment_from_utterance(utterance))
                    if [dialog['conversationId'], movie_id] not in forbidden_pairs and not np.isnan(movie_sentiment):
                        line['movie_id'] = movie_id
                        line['score'] = movie_sentiment
                        line['text'] = get_context(dialog, utterance, sep=self.sep)
                        list_of_lines.append(line)
        trainset = pd.DataFrame.from_records(list_of_lines)[['text', 'score']]
        trainset.to_csv(path, sep='\t', index=False)
        return trainset

    def train_sentiment_model(self):
        ''' train a sentiment model with unused conversations '''

        path = self.get_path('estimator.pkl')
        if not self.regenerate and os.path.exists(path):
            return pickle.load(open(path, 'rb'))

        sentiment_df = self.create_sentiment_trainset()
        sentiment_embeddings = self.embed_text(
            sentiment_df['text'].to_list(),
            self.get_path('trainset_embeddings.txt')
        )

        # remove those few conversations which have score 3 to reduce noise
        indexes = sentiment_df[sentiment_df['score'] == 3].index.to_list()
        sentiment_df = sentiment_df[sentiment_df['score'] != 3]
        sentiment_embeddings = [i for ind, i in enumerate(sentiment_embeddings) if ind not in indexes]

        self.logger.info('Fitting sentiment estimator')
        x_train, x_test, y_train, y_test = tts(
            sentiment_embeddings,
            sentiment_df['score'],
            test_size=0.1,
            random_state=self.seed,
        )
        estimator = RFR(
            n_estimators=500,
            max_depth=10,
            n_jobs=-1,
            random_state=self.seed
        )
        estimator.fit(x_train, y_train)
        self.logger.info('Estimator fitted')

        pickle.dump(estimator, open(path, 'wb'))

        y_pred = estimator.predict(x_test)
        self.logger.info('sentiment estimator results:')
        self.logger.info(f'RMSE: {math.sqrt(mse(y_test, y_pred)):.4f}')
        self.logger.info(f' MAE: {mae(y_test, y_pred):.4f}')
        self.logger.info(f' R^2: {estimator.score(x_test, y_test):.4f}')

        return estimator

    def get_cf_feature(self):
        '''
        train a sentiment model with unused conversations
        estimate the sentiment for movies mentioned in the conversation using a sentiment model,
        append the conversation user's scores to the CF matrix and train the CF model
        predict the CF score for the unseen movie using CF model
        '''

        # estimate scores for unseen movies using CF model
        dialogs_w_cf_path = self.get_path('dialogs_w_cf.tsv')
        if not self.regenerate and os.path.exists(dialogs_w_cf_path):
            return pd.read_table(dialogs_w_cf_path)

        estimator = self.train_sentiment_model()
        embeddings = self.embed_text(
            self.dialogs['text_1'].tolist() + self.dialogs['text_2'].tolist(),
            self.get_path('dialogs_embeddings.txt')
        )

        # calculate the scores for movies in conversations using the estimator
        dialogs_estimated_path = self.get_path('dialogs_estimated.tsv')
        if not self.regenerate and os.path.exists(dialogs_estimated_path):
            self.dialogs = pd.read_table(dialogs_estimated_path)
        else:
            self.dialogs['score_1'] = estimator.predict(embeddings[:self.dialogs.shape[0]])
            self.dialogs['score_2'] = estimator.predict(embeddings[self.dialogs.shape[0]:])
            self.dialogs.drop(['text_1', 'text_2'], axis=1, inplace=True)
            self.dialogs.to_csv(dialogs_estimated_path, sep='\t', index=False)

        self.cf = self.train_cf()

        self.dialogs['cf_score'] = self.dialogs.apply(lambda x: self.cf.predict(x['critic_id'], x['movie_3']).est, axis=1)
        self.logger.info(f"RMSE for CF: {math.sqrt(mse(self.dialogs['cf_score'], self.dialogs['score_3'])):.4f}")
        self.logger.info(f" MAE for CF: {mae(self.dialogs['cf_score'], self.dialogs['score_3']):.4f}")
        self.dialogs.to_csv(dialogs_w_cf_path, sep='\t', index=False)
        return self.dialogs

    def get_metadata_features(self, df: pd.DataFrame):

        path = self.get_path('metadata_features.tsv')
        if not self.regenerate and os.path.exists(path):
            return pd.read_table(path)

        feats = []
        for (_, row), user in tqdm(
            zip(df.iterrows(), self.u_vectors),
            desc='metadata features',
            total=df.shape[0],
            dynamic_ncols=True,
        ):
            feats.append(self.get_single_meta(row, user))

        feats_df = pd.DataFrame.from_records(feats)
        feats_df.to_csv(path, index=False, sep='\t')
        return feats_df

    def get_single_meta(self, row, user):

        current_movie_features = self.movies_features[row['movie_3']]
        for feature in ['genre', 'people', 'description', 'title']:
            if feature not in current_movie_features or not current_movie_features[feature]:
                self.logger.warning(f"no {feature} for {row['movie_3']}")
        if not ('audience_score' in current_movie_features and 'critic_score' in current_movie_features):
            self.logger.warning('no scores for ', row['movie_3'])
            row['meta_audience_score'] = np.nan
            row['meta_critics_score'] = np.nan
        else:
            row['meta_audience_score'] = current_movie_features['audience_score'] / 100
            row['meta_critics_score'] = current_movie_features['critic_score'] / 100

        to_encode = [
            ', '.join(current_movie_features['genre']),
            ', '.join(current_movie_features['people'][:10]),
            current_movie_features['description'],
            current_movie_features['title']
        ]
        for i in range(len(to_encode)):
            if len(to_encode[i]) == 0:
                to_encode[i] = 'None'
        genre, people, description, title = self.bc.encode(to_encode, show_progress_bar=False, batch_size=1024)

        row['meta_date'] = current_movie_features['in theaters']
        row['meta_runtime'] = current_movie_features['runtime']
        row['meta_amount_critics'] = current_movie_features['amount of critics']
        row['meta_amount_users'] = current_movie_features['amount of users']
        row['meta_genre'] = np.dot(user, np.transpose(genre))
        row['meta_people'] = np.dot(user, np.transpose(people))
        row['meta_description'] = np.dot(user, np.transpose(description))
        row['meta_title'] = np.dot(user, np.transpose(title))

        return row
