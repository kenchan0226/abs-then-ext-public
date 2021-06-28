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


@curry
def process(split, i):
    data_dir = join(DATA_DIR, split)
    with open(join(data_dir, '{}.json'.format(i))) as f:
        data = json.loads(f.read())
    tokenize = compose(list, _split_words)
    art_sents = tokenize(data['article'])
    abs_sents = tokenize(data['abstract'])
    ext_labels = data['extracted']
    compression_ratios = []
    if art_sents and abs_sents:  # some data contains empty article/abstract
        for ext_label, abs_sent in zip(ext_labels, abs_sents):
            original_sent = art_sents[ext_label]
            if len(original_sent) == 0:
                print(i)
                print(ext_label)
                print(art_sents[ext_label])
                print(abs_sent)
                exit()
                compression_ratio = 0.0
            else:
                compression_ratio = (len(original_sent) - len(abs_sent)) / len(original_sent)
            compression_ratios.append(compression_ratio)
    else:
        compression_ratios = []
    data['compression_ratios'] = compression_ratios
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
        for split in ['val', 'train']:
            #label_mp(split)
            label(split)
    else:
        #label_mp(split)
        label(split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Make extraction labels')
    )
    parser.add_argument('--split', type=str, action='store', default='all',
                        help='The folder name that needs to produce candidates. all means process both train and val.')
    args = parser.parse_args()
    main(args.split)
