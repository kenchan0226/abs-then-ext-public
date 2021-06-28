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


def _lcs_dp(a, b):
    """ compute the len dp of lcs"""
    dp = [[0 for _ in range(0, len(b)+1)]
          for _ in range(0, len(a)+1)]
    # dp[i][j]: lcs_len(a[:i], b[:j])
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp


def _lcs(a, b):
    """ compute the longest common subsequence between a and b"""
    dp = _lcs_dp(a, b)
    i = len(a)
    j = len(b)
    lcs = deque()
    while (i > 0 and j > 0):
        if a[i-1] == b[j-1]:
            lcs.appendleft(a[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    assert len(lcs) == dp[-1][-1]
    return lcs


def get_extract_label(art_sents, abs_sents, ROUGE_mode, ext_type, threshold):
    """ greedily match summary sentences to article sentences"""
    extracted = []
    #scores = []
    #indices = list(range(len(art_sents)))
    extracted_major = []
    distances = []
    for abst in abs_sents:
        #rouges = list(map(compute_rouge_l(reference=abst, mode='r'), art_sents))
        #rouges = list(map(compute_rouge_l(reference=abst, mode=ROUGE_mode), art_sents))  # Rouge-L F1
        ext, distance = get_extract_label_for_one_abstract_sent(art_sents, abst, ROUGE_mode, ext_type, threshold, extracted_major)
        #ext = max(indices, key=lambda i: rouges[i])
        #indices.remove(ext)
        extracted.append(ext)
        extracted_major.append(ext[0])
        distances.append(distance)
        #scores.append(rouges[ext])
        if len(extracted_major) == len(abs_sents):
            break
    return extracted, distances


def get_extract_label_for_one_abstract_sent(art_sents, abs_sent, ROUGE_mode, ext_type, threshold, escape_art_sent_indices=[]):
    # pick doc sentences with the highest Rouge-L recall (major sentence)
    #indices = list(range(len(art_sents)))
    #rouges = list(map(compute_rouge_l(reference=abs_sent, mode=ROUGE_mode), art_sents))  # Rouge-L F1
    rouges = [compute_rouge_l(output=art_sent, reference=abs_sent, mode=ROUGE_mode) if art_sent_idx not in escape_art_sent_indices else -1 for art_sent_idx, art_sent in enumerate(art_sents)]
    major_art_sent_rouge, major_art_sent_idx = find_max_val_and_idx(rouges)
    major_art_sent = art_sents[major_art_sent_idx]
    ext_labels = [major_art_sent_idx]
    distance = None

    if ext_type == 0:
        #threshold = 3
        major_lcs_len = _lcs_len(major_art_sent, abs_sent)
        #major_sent_idx = max(indices, key=lambda i: rouges[i])

        union_lcs_len_list = [get_union_lcs_len([major_art_sent, art_sent], abs_sent) if art_sent_idx != major_art_sent_idx else -1 for art_sent_idx, art_sent in enumerate(art_sents)]
        major_minor_union_lcs_len, minor_art_sent_idx = find_max_val_and_idx(union_lcs_len_list)
        if major_minor_union_lcs_len - major_lcs_len >= threshold:
            ext_labels.append(minor_art_sent_idx)
            distance = abs(major_art_sent_idx - minor_art_sent_idx)
    elif ext_type == 1:
        #threshold = 0.01
        minor_sent_candidates_rouges = [
            compute_rouge_l_summ([major_art_sent, art_sent], [abs_sent], mode=ROUGE_mode) if art_sent_idx != major_art_sent_idx else -1 for
            art_sent_idx, art_sent in enumerate(art_sents)]
        major_minor_rouge, minor_art_sent_idx = find_max_val_and_idx(minor_sent_candidates_rouges)
        if major_minor_rouge - major_art_sent_rouge >= threshold:
            ext_labels.append(minor_art_sent_idx)
            distance = abs(major_art_sent_idx - minor_art_sent_idx)

# compute_rouge_l_summ
    return ext_labels, distance


def find_max_val_and_idx(val_list):
    max_val = -1000000
    max_idx = -1
    for idx, val in enumerate(val_list):
        if val > max_val:
            max_val = val
            max_idx = idx
    return max_val, max_idx


def get_union_lcs_len(art_sents, ref_sent):
    tot_hit = 0
    art_cnt = Counter(concat(art_sents))
    ref_cnt = Counter(ref_sent)
    for art_sent in art_sents:
        lcs = _lcs(art_sent, ref_sent)
        for gram in lcs:
            if ref_cnt[gram] > 0 and art_cnt[gram] > 0:
                tot_hit += 1
            ref_cnt[gram] -= 1
            art_cnt[gram] -= 1
    return tot_hit


def _lcs_len(a, b):
    """ compute the length of longest common subsequence between a and b"""
    dp = _lcs_dp(a, b)
    return dp[-1][-1]


@curry
def process(split, ROUGE_mode, ext_type, threshold, i):
    data_dir = join(DATA_DIR, split)
    with open(join(data_dir, '{}.json'.format(i))) as f:
        data = json.loads(f.read())
    tokenize = compose(list, _split_words)
    art_sents = tokenize(data['article'])
    abs_sents = tokenize(data['abstract'])
    if art_sents and abs_sents: # some data contains empty article/abstract
        extracted, distances = get_extract_label(art_sents, abs_sents, ROUGE_mode, ext_type, threshold)
    else:
        extracted, distances = [], []
    #data.pop('extracted_by_lcs', None)
    #data.pop('distances_lcs', None)
    data['extracted_two_to_one_{}'.format(threshold)] = extracted
    data['distances_two_to_one_{}'.format(threshold)] = distances
    with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
        json.dump(data, f, indent=4)


def label_mp(split, ROUGE_mode, ext_type, threshold):
    """ process the data split with multi-processing"""
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    n_data = count_data(data_dir)
    with mp.Pool() as pool:
        list(pool.imap_unordered(process(split, ROUGE_mode, ext_type, threshold),
                                 list(range(n_data)), chunksize=1024))
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def main(split, ROUGE_mode, ext_type, threshold):
    if split == 'all':
        for split in ['val', 'train']:  # no need of extraction label when testing
            label_mp(split, ROUGE_mode, ext_type, threshold)
    else:
        label_mp(split, ROUGE_mode, ext_type, threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Make extraction labels')
    )
    parser.add_argument('--folder_name', type=str, action='store', default='all',
                        help='The folder name that needs to produce candidates. all means process both train and val.')
    parser.add_argument('--ROUGE_mode', type=str, action='store', default='r', choices=['r', 'f'],
                        help='The metric used to construct proxy extractive target label. r means Rouge-l recall. f means ROUGE-l F1.')
    parser.add_argument('--ext_type', type=int, action='store', default=0, choices=[0, 1],
                        help='0: use marginal increase in LCS. 1: use marginal increase in Rouge-L recall')
    parser.add_argument('--threshold', type=float, action='store',
                        help='Threshold for including a minor sentence.')
    args = parser.parse_args()
    main(args.folder_name, args.ROUGE_mode, args.ext_type, args.threshold)

