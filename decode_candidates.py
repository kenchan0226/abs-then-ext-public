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
import heapq

from toolz.sandbox.core import unzip
from cytoolz import concat, curry

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp

from data.batcher import tokenize

from decoding import Abstractor, BeamAbstractor
from data.data import CnnDmDataset, CnnDmDatasetFromIdx


try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


class DecodeCandidateDataset(CnnDmDatasetFromIdx):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split, start_idx=0):
        super().__init__(split, DATA_DIR, start_idx)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        abs_sents = js_data['abstract']
        return art_sents, abs_sents


def coll(batch):
    filtered_art_batch = []
    filtered_abs_batch = []
    empty_data_indices = []
    # filter out all empty articles
    for batch_i, (art_sents, abs_sents) in enumerate(batch):
        if art_sents:  # only keep non empty articles
            filtered_art_batch.append(art_sents)
            filtered_abs_batch.append(abs_sents)
        else:  # log the empty idx
            empty_data_indices.append(batch_i)
    return filtered_art_batch, filtered_abs_batch, empty_data_indices


def decode(save_path, abs_dir, split, batch_size, beam_size, diverse, max_len, topk, rerank_mode, keep_original_sent, start_idx, cuda, debug=False):
    start = time()
    # setup model
    assert abs_dir is not None
    if beam_size ==1:
        abstractor = Abstractor(abs_dir, max_len, cuda)
    else:
        abstractor = BeamAbstractor(abs_dir, max_len, cuda)

    # a dummy extractor that extract all the sentences
    extractor = lambda art_sents: list(range(len(art_sents)))

    dataset = DecodeCandidateDataset(split, start_idx)  # only need json['article'] and json['abstract']

    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll
    )

    # prepare save paths and logs
    os.makedirs(join(save_path, '{}_candidate'.format(split)))
    dec_log = {}
    dec_log['abstractor'] = (json.load(open(join(abs_dir, 'meta.json'))))
    dec_log['extractor'] = None
    dec_log['rl'] = False
    dec_log['split'] = split
    dec_log['beam'] = beam_size
    dec_log['diverse'] = diverse
    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)

    # Decoding
    i = 0
    with torch.no_grad():
        for i_debug, (raw_article_batch, raw_abs_batch, empty_data_indices) in enumerate(loader):
            if debug:
                print("raw article batch")
                print(raw_article_batch[0][0])
                print(raw_article_batch[0][1])
                print("article lengths")
                print([len(art) for art in raw_article_batch])
            tokenized_article_batch = map(tokenize(100, "w2v", None), raw_article_batch)
            ext_arts = []
            ext_inds = []
            for raw_art_sents in tokenized_article_batch:
                ext = extractor(raw_art_sents)
                ext_inds += [(len(ext_arts), len(ext))]
                ext_arts += list(map(lambda i: raw_art_sents[i], ext))
            if debug:
                print("ext_inds")
                print(ext_inds)
            if beam_size > 1:
                all_beams = abstractor(ext_arts, beam_size, diverse)  # a list of beam for the whole batch

                if debug:
                    print("Beams:")
                    for beam_i, beam in enumerate(all_beams):
                        print([hyp.sequence for hyp in beam])
                    print("======")
                dec_outs = rerank_mp(all_beams, ext_inds, topk, rerank_mode)
                #dec_outs = rerank(all_beams, ext_inds, topk, rerank_mode)
                # dec_outs: a list of list of token list
            else:
                dec_outs = abstractor(ext_arts)

            #assert i == batch_size * i_debug
            if i != batch_size * i_debug:
                print("i: {}".format(i))
                print("batch_size: {}, i_debug: {}, batch_size * i_debug: {}".format(batch_size, i_debug, batch_size * i_debug))
                raise ValueError

            if debug:
                print("dec_outs[0]")
                print(dec_outs[0])
                print("dec_outs[1]")
                print(dec_outs[1])
                print("length of dec_out")
                print(len(dec_outs))
                print("article output")

            """
            if i_debug == 18:
                print("Length of ext_ids: {}".format(len(ext_inds)))
                print("Length of raw_rticle_batch: {}".format(len(raw_article_batch)))
                print("Length of tokenized_article_batch: {}".format(len(list(tokenized_article_batch))))
                print("i: {}".format(i))
            """

            # insert place holders for samples with empty article
            for empty_idx in empty_data_indices:
                ext_inds.insert(empty_idx, (None, None))

            batch_i = 0
            for j, n in ext_inds:

                if debug:
                    print("j: {}, n: {}".format(j, n))

                # one article
                article_decoded_sents = []  # a list of all candidates sentences for one article

                if j is not None and n is not None:  # if the input article is not empty
                    # construct a list of all candidate sentences in one article, a list of str.
                    for sent_i, sent in enumerate(dec_outs[j:j + n]):
                        # one sent
                        if beam_size > 1:
                            candidate_list = [' '.join(candidate) for candidate in sent]
                        else:
                            candidate_list = [' '.join(sent)]

                        if keep_original_sent:
                            candidate_list.insert(0, raw_article_batch[batch_i][sent_i])

                        article_decoded_sents += candidate_list
                    # fetch the abstract of the original sample
                    raw_abstract = raw_abs_batch[batch_i]
                    batch_i += 1
                else:
                    raw_abstract = []

                    if debug:
                        print(article_decoded_sents[0])
                        print(article_decoded_sents[1])
                        print(article_decoded_sents[2])
                        print(article_decoded_sents[3])

                json_dict = {"article": article_decoded_sents, "abstract": raw_abstract}

                with open(join(save_path, '{}_candidate/{}.json'.format(split, i+start_idx)),
                          'w') as f:
                    f.write(json.dumps(json_dict))
                i += 1

                """
                if i_debug == 18:
                    art_len = len(json_dict['abstract'])
                    print("length of current article: {}".format(art_len))
                    if art_len > 0:
                        print(json_dict['abstract'][0])
                    print("i increases to: {}\n".format(i))
                """

                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                    i, n_data, i / n_data * 100,
                    timedelta(seconds=int(time() - start))
                ), end='')

                if debug:
                    raise ValueError
            """
            if i_debug == 18:
                raise ValueError
            """
    print()


_PRUNE = defaultdict(
    lambda: 2,
    {1:5, 2:5, 3:5, 4:5, 5:5, 6:4, 7:3, 8:3}
)


def rerank(all_beams, ext_inds, k, rerank_mode=0):
    beam_lists = (all_beams[i: i+n] for i, n in ext_inds if n > 0)
    # a list of beam list, each beam list contains the beam for one article
    if rerank_mode == 0:
        topked = map(topk_one(k=k), beam_lists)
    elif rerank_mode == 1:
        topked = map(rerank_topk_one(k=k), beam_lists)
    else:
        topked = map(rerank_by_length(k=k), beam_lists)
    return list(concat(topked))


def rerank_mp(all_beams, ext_inds, k, rerank_mode=0):
    beam_lists = [all_beams[i: i+n] for i, n in ext_inds if n > 0]
    # a list of beam list, each beam list contains the beams for one article
    with mp.Pool(8) as pool:
        if rerank_mode == 0:
            topked = pool.map(topk_one(k=k), beam_lists)
        elif rerank_mode == 1:
            topked = pool.map(rerank_topk_one(k=k), beam_lists)
        elif rerank_mode == 2:
            topked = pool.map(rerank_by_length(k=k), beam_lists)
        else:
            raise ValueError
    return list(concat(topked))  # a list contains the candidates sentences for all articles in the batch


@curry
def rerank_by_length(beams, k):
    """
    :param beams: a list of beam in one article
    :return:
    """
    # sort it according to lengths, longest sequence first.
    #beams = [beam.sort(key=lambda h: -len(h.sequence)) for beam in beams]
    beams = [sorted(beam, key=lambda h: -len(h.sequence)) for beam in beams]
    art_dec_outs = []
    for beam in beams:
        # append the candidates for each beam
        beam_size = len(beam)
        if k == 2:
            art_dec_outs.append([beam[0].sequence, beam[-1].sequence])
        elif k == 3:
            art_dec_outs.append([beam[0].sequence, beam[beam_size//2].sequence, beam[-1].sequence])
        else:
            raise ValueError
    return art_dec_outs


def _compute_len_score(hyp):
    return len(hyp.sequence), hyp.logprob


@curry
def rerank_topk_one(beams, k):
    """
    :param beams: a list of beam in one article
    :param k:
    :return: art_dec_outs: a list of list of token list, len(art_dec_outs)=num_sents_in_article, len(art_dec_outs[0])=num_cands_in_sent_0
    """
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]
    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
    beams_with_topk_hyps = [heapq.nlargest(k, hyps, key=_compute_score) for hyps in beams]
    art_dec_outs = []
    for topk_hyps in beams_with_topk_hyps:
        art_dec_outs.append([h.sequence for h in topk_hyps])
    return art_dec_outs


@curry
def topk_one(beams, k):
    # beams: a list of beam in one article
    art_dec_outs = []  # a list of token list for an article, each token list is a candidate sentence
    for hyps in beams:  # hypotheses for each input sentence
        sent_candidates = [h.sequence for h in hyps[:k]]
        art_dec_outs.append(sent_candidates)
    return art_dec_outs


def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))


def _compute_score(hyp):
    repeat = sum(c-1 for g, c in hyp.gram_cnt.items() if c > 1)
    lp = hyp.logprob / len(hyp.sequence)
    return (-repeat, lp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('decode a pretrained abstractor to generate candidates')
    )
    parser.add_argument('--path', required=True, help='path to store/eval')
    parser.add_argument('--abs_dir', help='root of the abstractor model')

    # dataset split
    #data = parser.add_mutually_exclusive_group(required=True)
    #data.add_argument('--val', action='store_true', help='use validation set')
    #data.add_argument('--test', action='store_true', help='use test set')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='batch size of faster decoding')
    parser.add_argument('--beam', type=int, action='store', default=1,
                        help='beam size for beam-search (reranking included)')
    parser.add_argument('--div', type=float, action='store', default=1.0,
                        help='diverse ratio for the diverse beam-search')
    parser.add_argument('--max_dec_word', type=int, action='store', default=30,
                        help='maximun words to be decoded for the abstractor')

    parser.add_argument('--topk', type=int, action='store', default=2,
                        help='number of candidates to generate for each sentence')
    #parser.add_argument('--final_rerank', action='store_true',
    #                    help='rereank the output of diverse beam search by n-gram repeat')
    parser.add_argument('--rerank_mode', type=int, action='store', default=0, choices=[0,1,2],
                        help='0: no rerank, 1: rerank by repeat 2-gram, 2: rerank by length')
    parser.add_argument('--remove_original_sentence', action='store_true',
                        help='remove the original sentence from the candidates')
    parser.add_argument('--debug', action='store_true',
                        help='debug')

    parser.add_argument('--start_idx', type=int, action='store', default=0,
                        help='Read the data from the specified index.')

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')

    parser.add_argument('--split', type=str, action='store', default='train',
                        help='The split that needs to produce candidates.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    keep_original_sentence = not args.remove_original_sentence

    decode(args.path, args.abs_dir,
           args.split, args.batch, args.beam, args.div,
           args.max_dec_word, args.topk, args.rerank_mode, keep_original_sentence, args.start_idx, args.cuda, args.debug)
