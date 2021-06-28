""" run decoding of X-ext (+ abs)"""
import argparse
import json
import os
from os.path import join
from datetime import timedelta
from time import time
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op

from cytoolz import identity, concat, curry

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp

from data.batcher import tokenize

from decoding import Abstractor, Extractor, DecodeDataset, BeamAbstractor, ConditionalAbstractor, BeamConditionalAbstractor
from decoding import make_html_safe


MAX_ABS_NUM = 6  # need to set max sentences to extract for non-RL extractor


def decode(save_path, abs_dir, ext_dir, split, batch_size, max_len, beam_size, diverse, cuda, disable_final_rerank=False, is_conditional_abs=False):
    start = time()
    # setup model
    if abs_dir is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        abstractor = identity
    elif is_conditional_abs:
        if beam_size == 1:
            abstractor = ConditionalAbstractor(abs_dir, max_len, cuda)
        else:
            abstractor = BeamConditionalAbstractor(abs_dir, max_len, cuda)
    else:
        #abstractor = Abstractor(abs_dir, max_len, cuda)
        if beam_size == 1:
            abstractor = Abstractor(abs_dir, max_len, cuda)
        else:
            abstractor = BeamAbstractor(abs_dir, max_len, cuda)

    if ext_dir is None:
        # NOTE: if no abstractor is provided then
        #       it would be  the lead-N extractor
        extractor = lambda art_sents: list(range(len(art_sents)))[:MAX_ABS_NUM]
    else:
        extractor = Extractor(ext_dir, max_ext=MAX_ABS_NUM, cuda=cuda)

    # setup loader
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles
    dataset = DecodeDataset(split)

    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll
    )

    # prepare save paths and logs
    for i in range(MAX_ABS_NUM):
        os.makedirs(join(save_path, 'output_{}'.format(i)))
    dec_log = {}
    dec_log['abstractor'] = (None if abs_dir is None
                             else json.load(open(join(abs_dir, 'meta.json'))))
    dec_log['extractor'] = (None if ext_dir is None
                            else json.load(open(join(ext_dir, 'meta.json'))))
    dec_log['rl'] = False
    dec_log['split'] = split
    dec_log['beam'] = beam_size
    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)

    # Decoding
    i = 0
    with torch.no_grad():
        for i_debug, raw_article_batch in enumerate(loader):
            tokenized_article_batch = map(tokenize(None), raw_article_batch)
            num_articles = len(raw_article_batch)
            num_ext_sents = 0
            if is_conditional_abs:
                sequential_ext_sents = []
                sequential_article_ids = []
            else:
                ext_arts = []
            ext_inds = []
            for article_i, raw_art_sents in enumerate(tokenized_article_batch):
                ext = extractor(raw_art_sents)
                ext_inds += [(num_ext_sents, len(ext))]
                num_ext_sents += len(ext)

                if is_conditional_abs:
                    # insert place holder to sequential_ext_sents
                    num_selected_sents_excluded_eos = len(ext)
                    if num_selected_sents_excluded_eos > len(sequential_ext_sents):
                        [sequential_ext_sents.append([]) for _ in
                         range(num_selected_sents_excluded_eos - len(sequential_ext_sents))]
                        [sequential_article_ids.append([]) for _ in
                         range(num_selected_sents_excluded_eos - len(sequential_article_ids))]

                    for idx_i, idx in enumerate(ext):
                        sequential_ext_sents[idx_i].append(raw_art_sents[idx])
                        sequential_article_ids[idx_i].append(article_i)
                else:
                    ext_arts += list(map(lambda i: raw_art_sents[i], ext))

            if beam_size > 1:
                if is_conditional_abs:
                    dec_outs = abstractor(sequential_ext_sents, sequential_article_ids, num_articles, beam_size,
                                          diverse)
                else:
                    all_beams = abstractor(ext_arts, beam_size, diverse)  # a list of beam for the whole batch
                    dec_outs = rerank_mp(all_beams, ext_inds, disable_final_rerank)
            else:
                if is_conditional_abs:
                    dec_outs = abstractor(sequential_ext_sents, sequential_article_ids, num_articles)
                else:
                    dec_outs = abstractor(ext_arts)
            assert i == batch_size*i_debug
            for j, n in ext_inds:
                decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
                for k, dec_str in enumerate(decoded_sents):
                    with open(join(save_path, 'output_{}/{}.dec'.format(k, i)),
                              'w') as f:
                        f.write(make_html_safe(dec_str))

                i += 1
                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                    i, n_data, i/n_data*100, timedelta(seconds=int(time()-start))
                ), end='')
    print()

_PRUNE = defaultdict(
    lambda: 2,
    {1:5, 2:5, 3:5, 4:5, 5:5, 6:4, 7:3, 8:3}
)

def rerank(all_beams, ext_inds, disable_final_rerank=False):
    final_rerank = not disable_final_rerank
    beam_lists = (all_beams[i: i+n] for i, n in ext_inds if n > 0)
    if final_rerank:
        return list(concat(map(rerank_one, beam_lists)))
    else:
        return list(concat(map(top_one, beam_lists)))

def rerank_mp(all_beams, ext_inds, disable_final_rerank=False):
    final_rerank = not disable_final_rerank
    beam_lists = [all_beams[i: i+n] for i, n in ext_inds if n > 0]
    # a list of beam list, each beam list contains the beam for one article
    with mp.Pool(8) as pool:
        if final_rerank:
            reranked = pool.map(rerank_one, beam_lists)
        else:
            reranked = pool.map(top_one, beam_lists)
    return list(concat(reranked))

def rerank_one(beams):
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]
    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
    best_hyps = max(product(*beams), key=_compute_score)  # a tuple of
    dec_outs = [h.sequence for h in best_hyps]
    return dec_outs

def top_one(beams):
    return [hyps[0].sequence for hyps in beams]

def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))

def _compute_score(hyps):
    all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
    repeat = sum(c-1 for g, c in all_cnt.items() if c > 1)
    lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
    return (-repeat, lp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('combine an extractor and an abstractor '
                     'to decode summary and evaluate on the '
                     'CNN/Daily Mail dataset')
    )
    parser.add_argument('--path', required=True, help='path to store/eval')
    parser.add_argument('--abs_dir', help='root of the abstractor model')
    parser.add_argument('--ext_dir', help='root of the extractor model')

    # dataset split
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--val', action='store_true', help='use validation set')
    data.add_argument('--test', action='store_true', help='use test set')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='batch size of faster decoding')
    parser.add_argument('--n_ext', type=int, action='store', default=4,
                        help='number of sents to be extracted')
    parser.add_argument('--max_dec_word', type=int, action='store', default=30,
                        help='maximun words to be decoded for the abstractor')

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--is_conditional_abs', action='store_true', help='use conditional abstractor')

    parser.add_argument('--beam', type=int, action='store', default=1,
                        help='beam size of abstractor decoding (reranking included)')
    parser.add_argument('--div', type=float, action='store', default=1.0,
                        help='diverse ratio for the diverse beam-search')
    parser.add_argument('--disable_final_rerank', action='store_true',
                        help='disable the final rerank of the beam search output by n-gram repeat')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    data_split = 'test' if args.test else 'val'
    decode(args.path, args.abs_dir, args.ext_dir,
           data_split, args.batch, args.max_dec_word, args.beam, args.div, args.cuda, args.disable_final_rerank, args.is_conditional_abs)
