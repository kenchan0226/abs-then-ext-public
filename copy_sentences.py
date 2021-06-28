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


@curry
def process(src_folder, trg_folder, out_folder, k_src, k_trg, copy_i, insert_i, i):
    src_data_dir = join(DATA_DIR, src_folder)
    trg_data_dir = join(DATA_DIR, trg_folder)
    out_data_dir = join(DATA_DIR, out_folder)
    with open(join(src_data_dir, '{}.json'.format(i))) as f:
        src_data = json.loads(f.read())
    with open(join(trg_data_dir, '{}.json'.format(i))) as f:
        trg_data = json.loads(f.read())
    src_art_sents = src_data['article']
    trg_art_sents = trg_data['article']
    trg_abs_sents = trg_data['abstract']
    assert len(src_art_sents) % k_src == 0
    assert len(trg_art_sents) % k_trg == 0
    num_original_sents = len(trg_art_sents) // k_trg
    appended_trg_art_sents = []
    for sent_i in range(num_original_sents):
        trg_local_sent_i = sent_i * k_trg
        src_local_sent_i = sent_i * k_src
        trg_sent_cands = trg_art_sents[trg_local_sent_i:trg_local_sent_i + k_trg]
        src_sent_cands = src_art_sents[src_local_sent_i:src_local_sent_i + k_src]
        trg_sent_cands.insert(insert_i, src_sent_cands[copy_i])
        appended_trg_art_sents += trg_sent_cands

    new_json_dict = {"article": appended_trg_art_sents, "abstract": trg_abs_sents}
    with open(join(out_data_dir, '{}.json'.format(i)), 'w') as f:
        json.dump(new_json_dict, f, indent=4)


def copy_mp(src_folder, trg_folder, out_folder, k_src, k_trg, copy_i, insert_i):
    start = time()
    print('start processing {} split...'.format(trg_folder))
    trg_data_dir = join(DATA_DIR, trg_folder)
    n_data = count_data(trg_data_dir)
    # make output folder
    out_data_dir = join(DATA_DIR, out_folder)
    os.makedirs(out_data_dir)
    process(src_folder, trg_folder, out_folder, k_src, k_trg, copy_i, insert_i, 1000)

    with mp.Pool() as pool:
        list(pool.imap_unordered(process(src_folder, trg_folder, out_folder, k_src, k_trg, copy_i, insert_i),
                                 list(range(n_data)), chunksize=1024))
    print('finished in {}'.format(timedelta(seconds=time() - start)))


def main(src_folder, trg_folder, out_folder, k_src, k_trg, copy_i, insert_i):
    copy_mp(src_folder, trg_folder, out_folder, k_src, k_trg, copy_i, insert_i)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Make extraction labels')
    )
    parser.add_argument('--src_folder', type=str, action='store',
                        help='Source folder name.')
    parser.add_argument('--trg_folder', type=str, action='store',
                        help='Target folder name.')
    parser.add_argument('--out_folder', type=str, action='store',
                        help='Output folder name.')
    parser.add_argument('--k_src', type=int, action='store',
                        help='Number of candidates per sentencein the source document')
    parser.add_argument('--k_trg', type=int, action='store',
                        help='Number of candidates per sentencein the target document')
    parser.add_argument('--copy_i', type=int, action='store',
                        help='The i-th candidate to copy')
    parser.add_argument('--insert_i', type=int, action='store',
                        help='Insert to the i-th position')
    args = parser.parse_args()
    main(args.src_folder, args.trg_folder, args.out_folder, args.k_src, args.k_trg, args.copy_i, args.insert_i)
