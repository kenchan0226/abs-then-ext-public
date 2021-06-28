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

from cytoolz import identity, concat, curry

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp

from data.batcher import tokenize

from decoding import Abstractor, Extractor, DecodeDataset, BeamAbstractor
from decoding import make_html_safe


MAX_ABS_NUM = 6  # need to set max sentences to extract for non-RL extractor


def decode(save_path, abs_dir, ext_dir, split, batch_size, max_len, num_candidates, beam_size, diverse,
           final_rerank, keep_original_sent, cuda, abstracted, debug=False):
    start = time()
    # setup model
    assert beam_size >= num_candidates and num_candidates > 1


    if abstracted:
        abstractor = identity
    else:
        if beam_size == 1:
            abstractor = Abstractor(abs_dir, max_len, cuda)
        else:
            abstractor = BeamAbstractor(abs_dir, max_len, cuda)

    extractor = Extractor(ext_dir, max_ext=MAX_ABS_NUM, cuda=cuda)

    emb_type = extractor.emb_type
    print("emb_type")
    print(emb_type)

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
    dec_log['abstractor'] = json.load(open(join(abs_dir, 'meta.json')))
    dec_log['extractor'] = json.load(open(join(ext_dir, 'meta.json')))
    dec_log['rl'] = False
    dec_log['split'] = split
    dec_log['beam'] = beam_size
    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)

    # Decoding
    i = 0
    with torch.no_grad():
        for i_debug, raw_article_batch in enumerate(loader):
            #tokenized_article_batch = list(map(tokenize(None), raw_article_batch))
            tokenized_article_batch = list(map(tokenize(None, emb_type, num_candidates), raw_article_batch))

            art_ids = []
            if abstracted:
                num_passed_sents = 0
                for art_sents in tokenized_article_batch:
                    art_ids += [(num_passed_sents, len(art_sents))]
                    num_passed_sents += len(art_sents)
            else:
                tokenized_article_batch_flattened = []  # a list of tokenized sentence for all the articles in the batch
                for art_sents in tokenized_article_batch:
                    art_ids += [(len(tokenized_article_batch_flattened), len(art_sents))]
                    tokenized_article_batch_flattened += art_sents

                if beam_size > 1:
                    all_beams = abstractor(tokenized_article_batch_flattened, beam_size, diverse)  # a list of beam for the whole batch
                    dec_outs = rerank_mp(all_beams, art_ids, num_candidates - 1, final_rerank=final_rerank)
                    # dec_outs: a list of list of token list [total number of sentences in batch, num_candidates, seq_len]
                else:
                    dec_outs = abstractor(tokenized_article_batch_flattened)

            tokenized_article_batch_flattened = []

            assert i == batch_size * i_debug

            if debug:
                print("dec_outs[0]")
                print(dec_outs[0])
                print("dec_outs[1]")
                print(dec_outs[1])
                print("length of dec_out")
                print(len(dec_outs))
                print("article output")

            for batch_i, (j, n) in enumerate(art_ids):
                # one article
                raw_article_sents = raw_article_batch[batch_i]
                if abstracted:
                    art_sents_with_cands = tokenized_article_batch[batch_i]
                else:
                    art_sents_with_cands = []  # a list of tokenized sentence candidates for one article

                    for sent_i, sent in enumerate(dec_outs[j:j + n]):
                        # one sent
                        if beam_size > 1:
                            candidate_list = sent
                        else:
                            candidate_list = [sent]

                        if keep_original_sent:
                            candidate_list.insert(0, tokenized_article_batch[batch_i][sent_i])

                        art_sents_with_cands += candidate_list

                if debug:
                    print("art_sents_with_cands: {}".format(' '.join(art_sents_with_cands[0])))
                    print("art_sents_with_cands: {}".format(' '.join(art_sents_with_cands[1])))
                    print("art_sents_with_cands: {}".format(' '.join(art_sents_with_cands[2])))
                    print("art_sents_with_cands: {}".format(' '.join(art_sents_with_cands[3])))

                # extraction
                ext = extractor(art_sents_with_cands)

                if debug:
                    print("ext: {}".format(ext))

                # write to .dec
                for k, ext_i in enumerate(ext):
                    #ext_str = ' '.join(art_sents_with_cands[ext_i])
                    ext_str = raw_article_sents[ext_i]
                    if debug:
                        print("k: {}, i: {}, sent: {}".format(k, i, ext_str))
                    with open(join(save_path, 'output_{}/{}.dec'.format(k, i)),
                              'w') as f:
                        f.write(make_html_safe(ext_str))
                i += 1
                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                    i, n_data, i / n_data * 100, timedelta(seconds=int(time() - start))
                ), end='')
                if debug:
                    raise ValueError

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
        description=('combine an extractor and an abstractor '
                     'to decode summary and evaluate on the '
                     'CNN/Daily Mail dataset')
    )
    parser.add_argument('--path', required=True, help='path to store/eval')
    parser.add_argument('--abs_dir', help='root of the abstractor model', required=True)
    parser.add_argument('--ext_dir', help='root of the extractor model', required=True)

    # dataset split
    #data = parser.add_mutually_exclusive_group(required=True)
    #data.add_argument('--val', action='store_true', help='use validation set')
    #data.add_argument('--test', action='store_true', help='use test set')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='batch size of faster decoding')
    parser.add_argument('--n_ext', type=int, action='store', default=4,
                        help='number of sents to be extracted')
    parser.add_argument('--max_dec_word', type=int, action='store', default=30,
                        help='maximun words to be decoded for the abstractor')

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')

    parser.add_argument('--beam', type=int, action='store', default=1,
                        help='beam size of abstractor decoding (reranking included)')
    parser.add_argument('--div', type=float, action='store', default=1.0,
                        help='diverse ratio for the diverse beam-search')
    parser.add_argument('--num_candidates', type=int, action='store', default=2,
                        help='The number of candidates for each sentence.')
    parser.add_argument('--final_rerank', action='store_true',
                        help='rereank the output of diverse beam search by n-gram repeat')

    parser.add_argument('--remove_original_sentence', action='store_true',
                        help='remove the original sentence from the candidates')
    parser.add_argument('--test_set_folder', type=str, action='store', default="test",
                        help='The name of testing set folder')
    parser.add_argument('--abstracted', action='store_true',
                        help='Inidcate the test set already been abstracted, so the agent will not do the abstraction anymore')
    parser.add_argument('--debug', action='store_true',
                        help='debug')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    keep_original_sentence = not args.remove_original_sentence

    #data_split = 'test' if args.test else 'val'
    decode(args.path, args.abs_dir, args.ext_dir,
           args.test_set_folder, args.batch, args.max_dec_word, args.num_candidates, args.beam, args.div,
           args.final_rerank, keep_original_sentence, args.cuda, args.abstracted, args.debug)
