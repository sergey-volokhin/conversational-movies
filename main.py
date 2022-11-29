import argparse
import logging
import os
import sys

from dataset import MyDataset
from model import MyModel


def get_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger()


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--regenerate',
        help='force regenerate all data, embeddings and model',
        action='store_true'
    )
    parser.add_argument(
        '-d', '--directory',
        type=str,
        help='directory where everything is saved',
        default='new_tmp'
    )
    parser.add_argument(
        '-f', '--feature_importances',
        help='whether to print feature importances of the model',
        action='store_true'
    )
    parser.add_argument(
        '-cf', '--cf_type',
        type=str,
        help='which cf model to use',
        choices=['svd', 'svdpp', 'knn'],
        default='knn'
    )
    parser.add_argument(
        '-b', '--bert-model',
        type=str,
        default='all-mpnet-base-v2',
        help='''
            which pretrained SentenceTransformer model to use
            (e.g. "all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1", etc)
        '''
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    args = parser.parse_args()
    args.outpath = os.path.join(
        os.path.dirname(os.path.abspath(sys.argv[0])),
        args.directory
    )
    return args


if __name__ == '__main__':
    args = parse_args()
    args.logger = get_logger()
    dataset = MyDataset(args)
    dataset.load_all_features()
    model = MyModel(args)
    model.fit(dataset)
