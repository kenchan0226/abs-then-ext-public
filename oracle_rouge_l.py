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
def process(in_folder, out_folder, i):
    #tokenize = compose(list, _split_words)
    in_data_dir = join(DATA_DIR, in_folder)
    out_data_dir = join(out_folder, 'output')
    with open(join(in_data_dir, '{}.json'.format(i))) as f:
        data = json.loads(f.read())
    art_sents = data['article']
    abs_sents = data['abstract']
    exts = data['extracted']
    oracle_sents = [art_sents[ext] for ext in exts]
    with open(join(out_data_dir, '{}.dec'.format(i)), 'w') as f:
        f.writelines(oracle_sents)


def remove_mp(in_folder, out_folder):
    """ process the data split with multi-processing"""
    start = time()
    print('start processing {} split...'.format(in_folder))
    in_data_dir = join(DATA_DIR, in_folder)
    n_data = count_data(in_data_dir)
    # make output folder
    #out_data_dir = join(DATA_DIR, out_folder)
    out_data_dir = join(out_folder, 'output')
    os.makedirs(out_data_dir)
    log_dict = {"split": in_folder}
    with open(join(out_folder, 'log.json'), 'w') as f:
        json.dump(log_dict, f, indent=4)

    with mp.Pool() as pool:
        list(pool.imap_unordered(process(in_folder, out_folder),
                                 list(range(n_data)), chunksize=1024))
    print('finished in {}'.format(timedelta(seconds=time() - start)))


def main(in_folder, out_folder):
    remove_mp(in_folder, out_folder)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Make extraction labels')
    )
    parser.add_argument('--in_folder', type=str, action='store',
                        help='The folder name that needs to remove sentences.')
    parser.add_argument('--out_folder', type=str, action='store',
                        help='Output folder name.')

    args = parser.parse_args()
    main(args.in_folder, args.out_folder)
