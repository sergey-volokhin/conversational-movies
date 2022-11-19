import json
import logging
import os
import re
import sys
import time
import traceback
import random

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests import TooManyRedirects
from tqdm import tqdm
from whoosh import fields, index

datapath = os.path.dirname(os.path.abspath(sys.argv[0])) + '/data/'
movie_ids = json.load(open(datapath + 'films_rt_ids.json', 'r'))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()

schema = fields.Schema(movie_id=fields.KEYWORD(stored=True, scorable=True),
                       critic_id=fields.KEYWORD(stored=True, scorable=True),
                       score=fields.NUMERIC(stored=True),
                       review=fields.TEXT(stored=True),
                       freshness=fields.KEYWORD(scorable=True))

reverse_scores = {1: ['*', 'E+', 'E', 'E-', 'F+', 'F', 'F-'],
                  2: ['**', 'D+', 'D PLUS', 'D', 'D-'],
                  3: ['***', 'C PLUS', 'C-PLUS', 'C+', 'C', 'C-', 'C-MINUS', 'C MINUS'],
                  4: ['****', 'B PLUS', 'B-PLUS', 'B +', 'B+', 'B', 'B-', 'B MINUS', 'B-MINUS'],
                  5: ['*****', 'A-PLUS', 'A PLUS', 'A+', 'A', 'A-', 'A -', 'A MINUS', 'A-MINUS']}
score_map = {s: k for k in reverse_scores for s in reverse_scores[k]}
crawl_rate = 1

rotten_main = 'https://www.rottentomatoes.com/'


def calculate_score(score):
    if score is None or score.strip() == '':
        return np.nan
    score = re.sub(' +', ' ', score.strip())
    try:
        res = float(eval(score.replace('\'', '').replace('"', '').replace(' stars out of ', '/').replace(' stars', '/5').replace(' out of ', '/').replace(' of ', '/')))
    except Exception:
        try:
            return score_map[score.upper()]
        except KeyError:
            return np.nan
    else:
        if 0 <= res <= 1:
            return max(1, round(res*5))
        return np.nan


def make_soup(url):
    time.sleep(crawl_rate)
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')
    except TooManyRedirects:
        soup = ''
    return soup


def get_critics_from_movie(page):
    address = f'{rotten_main}/m/{page}/reviews'
    soup = make_soup(address)
    try:
        page_nums = int(soup.find("span", class_='pageInfo').text[9:])
    except AttributeError:
        page_nums = 1

    critics = []
    for page_num in range(1, page_nums+1):
        page_soup = make_soup(address+f'?page={page_num}&sort=')
        reviews_soups = page_soup.find_all("div", class_="row review_table_row")
        for review_soup in reviews_soups:
            try:
                critics += [review_soup.find('a', class_='unstyled bold articleLink')['href'][8:]]
            except Exception:
                pass
    return critics


def get_reviews_for_critic(critic):
    address = f'{rotten_main}/napi/critic/{critic}/review/movie?offset='
    try:
        page = requests.get(address+'0').json()
        all_reviews_f_critic = []
        offset = 0
        total = page['totalCount']
        while offset < total:
            time.sleep(crawl_rate)
            page = requests.get(address+str(offset)).json()
            for review in page['results']:
                current_review = {'critic_id': critic}
                try:
                    current_review['movie_id'] = review['media']['url'][33:].replace('-', '_')
                except Exception:
                    continue
                current_review['fresh'] = review['score']
                current_review['score'] = review['scoreOri']
                current_review['review'] = review['quote']
                all_reviews_f_critic.append(current_review)
            offset += len(page['results'])
        return all_reviews_f_critic

    except Exception as err:
        logger.info("couldn't get reviews for {}. {}.".format(critic, err))
        traceback.logger.info_exc()
        open(datapath+'failed_critics', 'a+').write(critic+'\n')
        return []


def indexing(df, index_name=datapath+'index'):
    os.makedirs(index_name, exist_ok=True)
    ix = index.create_in(index_name, schema)
    writer = ix.writer()
    logger.info('Indexing reviews')
    for _, review in df.iterrows():
        writer.add_document(
            movie_id=review['movie_id'],
            critic_id=review['critic_id'],
            review=review['review'],
            score=review['score'])
    writer.commit()


if __name__ == '__main__':

    critics = []
    movie_ids = {k: v for k, v in random.sample(movie_ids.items(), 2)}
    for film in tqdm(movie_ids):
        critics += get_critics_from_movie(film)

    critics = sorted(list(set([i for i in critics if len(i)>0])))
    open(datapath+'critics_list.txt', 'w').write('\n'.join(critics))

    reviews = []
    open(datapath+'failed_critics', 'w').close()
    logger.info(f'Crawling reviews from {len(critics)} critics')
    for critic in tqdm(critics[:2]):
        reviews += get_reviews_for_critic(critic)

    failed_critics = open(datapath+'failed_critics', 'r').read().split('\n')
    open(datapath+'failed_critics', 'a+').write('\n')
    logger.info(f'Attempting to recrawl reviews from {len(failed_critics)} critics')
    for critic in failed_critics:
        if critic != '':
            reviews += get_reviews_for_critic(critic)

    df_all_reviews = pd.DataFrame.from_records(reviews).dropna()
    df_all_reviews['score'] = df_all_reviews['score'].apply(calculate_score)
    df_all_reviews = df_all_reviews.replace('', np.nan).dropna()

    # removing common templates
    df_all_reviews = df_all_reviews[df_all_reviews['movie_id'] != ':vanity']
    df_all_reviews['review_lower'] = df_all_reviews['review'].apply(lambda x: str(x).lower())
    df_all_reviews = df_all_reviews[~df_all_reviews['review_lower'].isin(['see website for more details.', '.'])]
    df_all_reviews = df_all_reviews[~df_all_reviews['review_lower'].str.startswith('click to ')]
    df_all_reviews = df_all_reviews[~df_all_reviews['review_lower'].str.startswith('click for ')]
    df_all_reviews = df_all_reviews[~df_all_reviews['review_lower'].str.startswith('full review ')]
    df_all_reviews.drop('review_lower', axis=1).to_csv(datapath + 'reviews.tsv', sep='\t', index=False)

    groupped = df_all_reviews.groupby('critic_id')
    logger.info(f"Total critics: {df_all_reviews['critic_id'].nunique()}")
    logger.info(f"Total reviews: {df_all_reviews.shape[0]}")
    logger.info(f"Number of movies: {len(df_all_reviews.groupby('movie_id'))}")
    logger.info(f'Median of reviews per critic: {np.median([len(i[1]) for i in groupped])}')
    logger.info(f'Mean of reviews per critic: {np.mean([len(i[1]) for i in groupped])}')

    indexing(df_all_reviews)
