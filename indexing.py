import os

import pandas as pd
from tqdm import tqdm
from whoosh import index

from model_utils import datapath, logger, schema, timeit


@timeit
def indexing(df, index_name=datapath+'index'):
    os.makedirs(index_name, exist_ok=True)
    ix = index.create_in(index_name, schema)
    writer = ix.writer()
    logger.info('Indexing reviews')
    for _, review in tqdm(df.iterrows(), total=df.shape[0]):
        writer.add_document(
            movie_id=review['movie_id'],
            critic_id=review['critic_id'],
            review=review['review'],
            score=review['score'])
    writer.commit()


if __name__ == '__main__':

    df = pd.read_table(datapath+'reviews.tsv.gz')
    indexing(df)
