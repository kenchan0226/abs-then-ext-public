"""Most of the codes are adapted from the source code in Chen et al. 2018"""

import os
from os.path import exists, join
import json
from time import time
from datetime import timedelta
import multiprocessing as mp

from cytoolz import concat, curry, compose

from utils import count_data
from metric import compute_rouge_l, compute_rouge_l_summ
import argparse
from collections import Counter, deque


try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


def _split_words(texts):
    return map(lambda t: t.split(), texts)


def check_empty(sents):
    if len(sents) == 1 and sents[0] == "":
        return True
    else:
        return False


@curry
def process(split, i):
    data_dir = join(DATA_DIR, split)
    with open(join(data_dir, '{}.json'.format(i))) as f:
        data = json.loads(f.read())
    art_sents = data['article']
    abs_sents = data['abstract']
    if check_empty(art_sents):
        data['article'] = []
        data['extracted'] = []
    if check_empty(abs_sents):
        data['abstract'] = []
        data['extracted'] = []
    with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
        json.dump(data, f, indent=4)


def label_mp(split):
    """ process the data split with multi-processing"""
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    n_data = count_data(data_dir)
    with mp.Pool() as pool:
        list(pool.imap_unordered(process(split),
                                 list(range(n_data)), chunksize=1024))
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def label(split):
    """ process the data split with multi-processing"""
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    n_data = count_data(data_dir)
    for i in range(n_data):
        process(split, i)
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def main(split):
    if split == 'all':
        for split in ['val', 'train', 'test']:
            label_mp(split)
            #label(split)
    else:
        label_mp(split)
        #label(split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Make extraction labels')
    )
    parser.add_argument('--split', type=str, action='store', default='all',
                        help='The folder name that needs to produce candidates. all means process both train and val.')
    args = parser.parse_args()
    main(args.split)
