import logging
import os

import pandas as pd
from tqdm import tqdm
from whoosh import fields, index, qparser, scoring


def create_index(df, index_name, schema):
    os.makedirs(index_name, exist_ok=True)
    ix = index.create_in(index_name, schema)
    writer = ix.writer()
    logging.info('Indexing reviews')
    for _, review in tqdm(df.iterrows(), total=df.shape[0], desc='adding reviews into index'):
        writer.add_document(
            movie_id=review['movie_id'],
            critic_id=review['critic_id'],
            review=review['review'],
            score=review['score'])
    logging.info('Committing reviews. This will take a while')
    writer.commit()
    return index.open_dir(index_name).searcher(weighting=scoring.BM25F)


def load_index(path, data):
    schema = fields.Schema(
        movie_id=fields.KEYWORD(stored=True, scorable=True),
        critic_id=fields.KEYWORD(stored=True, scorable=True),
        score=fields.NUMERIC(stored=True),
        review=fields.TEXT(stored=True),
        freshness=fields.KEYWORD(scorable=True)
    )
    whoosh_parser = qparser.MultifieldParser(['movie_id', 'review'], schema=schema)
    try:
        searcher = index.open_dir(os.path.join(path, 'index')).searcher(weighting=scoring.BM25F)
    except index.EmptyIndexError:
        searcher = create_index(data, path, schema)
    return whoosh_parser, searcher


if __name__ == '__main__':

    df = pd.read_table('reviews.tsv')
    create_index(df)
