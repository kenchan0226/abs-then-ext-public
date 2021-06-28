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
import io
from decoding import make_html_safe


try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


class TwoToOneMatchDataset(CnnDmDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split, threshold):
        super().__init__(split, DATA_DIR)
        self.threshold = threshold

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, abs_sents, extracts = (
            js_data['article'], js_data['abstract'], js_data['extracted_two_to_one_{}'.format(self.threshold)])
        # only keep the sentences with two ext labels
        matched_arts = []
        matched_abss = []
        for ext, abst in zip(extracts, abs_sents):
            if len(ext) == 2:
                primary_idx = ext[0]
                if primary_idx > 0:
                    secondary_idx = primary_idx - 1
                else:
                    secondary_idx = -1
                matched_arts.append( art_sents[primary_idx] + ' ' + art_sents[secondary_idx] )
            elif len(ext) == 1:
                matched_arts.append(art_sents[ext[0]])
            else:
                raise ValueError("Bug!")
            matched_abss.append(abst)
        # return matched_arts, abs_sents[:len(extracts)]
        return matched_arts, matched_abss


def coll(batch):
    filtered_art_batch = []
    filtered_abs_batch = []
    for batch_i, (art_sents, abs_sents) in enumerate(batch):
        filtered_art_batch.append(art_sents)
        filtered_abs_batch.append(abs_sents)
    return filtered_art_batch, filtered_abs_batch


def decode(save_path, abs_dir, split, batch_size, beam_size, diverse, max_len, final_rerank, cuda, debug=False):
    start = time()
    topk = 1

    # setup model
    assert abs_dir is not None
    if beam_size ==1:
        abstractor = Abstractor(abs_dir, max_len, cuda)
    else:
        abstractor = BeamAbstractor(abs_dir, max_len, cuda)

    # a dummy extractor that extract all the sentences
    extractor = lambda art_sents: list(range(len(art_sents)))

    dataset = TwoToOneMatchDataset(split, threshold=0.15)  # only need json['article'] and json['abstract']

    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll
    )

    # prepare save paths and logs
    os.makedirs(save_path)
    os.makedirs(join(save_path, 'output'))
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
        for i_debug, (raw_article_batch, raw_abs_batch) in enumerate(loader):
            if debug:
                print("raw article batch")
                print(raw_article_batch[0][0])
                print(raw_article_batch[0][1])
                print("article lengths")
                print([len(art) for art in raw_article_batch])
            # pick out the original sentence
            # raw_article_batch a list of list of sentences, article, then sentence in article
            #raw_original_article_batch = []
            #for raw_article_sents in raw_article_batch:
            #    original_article_sents = [article_sent for cand_i, article_sent in enumerate(raw_article_sents) if cand_i % exist_candidates == 0]
            #    raw_original_article_batch.append(original_article_sents)

            tokenized_original_article_batch = list(map(tokenize(None), raw_article_batch))
            if debug:
                print("tokenized_original_article_batch")
                print(tokenized_original_article_batch[0][0])
                print(tokenized_original_article_batch[0][1])

            ext_arts = []
            ext_inds = []
            for raw_art_sents in tokenized_original_article_batch:
                ext = extractor(raw_art_sents)
                ext_inds += [(len(ext_arts), len(ext))]
                ext_arts += list(map(lambda i: raw_art_sents[i], ext))
            if debug:
                print("ext_inds")
                print(ext_inds)
            if beam_size > 1:
                all_beams = abstractor(ext_arts, beam_size, diverse)  # a list of beam for the whole batch
                dec_outs = rerank_mp(all_beams, ext_inds, topk, final_rerank)
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
                print("dec_outs[2]")
                print(dec_outs[2])
                print("dec_outs[3]")
                print(dec_outs[3])
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

            batch_i = 0
            for j, n in ext_inds:

                if debug:
                    print("j: {}, n: {}".format(j, n))

                # one article
                article_decoded_sents = []  # a list of all candidate sentences for one article

                if j is not None and n is not None:  # if the input article is not empty

                    # construct a list of all candidate sentences in one article, a list of str.
                    for sent_i, sent in enumerate(dec_outs[j:j + n]):
                        candidate_list = []

                        # one sent
                        if beam_size > 1:
                            candidate_list += [' '.join(candidate) for candidate in sent]
                        else:
                            candidate_list += [' '.join(sent)]

                        #if keep_original_sent:
                        #    candidate_list.insert(0, raw_article_batch[batch_i][sent_i])

                        article_decoded_sents += candidate_list
                    # fetch the abstract of the original sample
                    raw_abstract = raw_abs_batch[batch_i]
                    batch_i += 1
                else:
                    raw_abstract = []

                if debug:
                    print("article_decoded_sents[0]")
                    for z in range(9):
                        print(article_decoded_sents[z])
                    print("article_decoded_sents len")
                    print(len(article_decoded_sents))

                with open(join(save_path, 'output', '{}.dec'.format(i)), 'w') as f:
                    f.write(make_html_safe('\n'.join(article_decoded_sents)))

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


def rerank(all_beams, ext_inds, k, final_rerank=False):
    beam_lists = (all_beams[i: i+n] for i, n in ext_inds if n > 0)
    # a list of beam list, each beam list contains the beam for one article
    if final_rerank:
        topked = map(rereank_topk_one(k=k), beam_lists)
    else:
        topked = map(topk_one(k=k), beam_lists)
    return list(concat(topked))


def rerank_mp(all_beams, ext_inds, k, final_rerank=False):
    beam_lists = [all_beams[i: i+n] for i, n in ext_inds if n > 0]
    # a list of beam list, each beam list contains the beam for one article
    with mp.Pool(8) as pool:
        if final_rerank:
            topked = pool.map(rereank_topk_one(k=k), beam_lists)
        else:
            topked = pool.map(topk_one(k=k), beam_lists)
    return list(concat(topked))  # a list contains the candidates sentences for all articles in the batch


@curry
def rereank_topk_one(beams, k):
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
    parser.add_argument('--final_rerank', action='store_true',
                        help='rereank the output of diverse beam search by n-gram repeat')
    parser.add_argument('--debug', action='store_true',
                        help='debug')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--split', type=str, action='store', default='train',
                        help='The split that needs to produce candidates.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    decode(args.path, args.abs_dir,
           args.split, args.batch, args.beam, args.div,
           args.max_dec_word, args.final_rerank, args.cuda, args.debug)
