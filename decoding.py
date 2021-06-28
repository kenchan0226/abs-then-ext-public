""" decoding utilities"""
import json
import re
import os
from os.path import join
import pickle as pkl
from itertools import starmap
from collections import Counter, defaultdict

from cytoolz import curry, concat

import torch

from utils import PAD, UNK, START, END
from model.copy_summ import CopySumm
from model.copy_cond_summ import CopyCondSumm
from model.controllable_abstractor import CompressControlSumm
from model.extract import ExtractSumm, PtrExtractSumm, PtrExtractRewrittenSumm, PtrExtractRewrittenBertSumm, PtrExtractRewrittenSentBertSumm, PtrExtractRewrittenSentWordBertSumm
from model.rl import ActorCritic, ActorCriticCand, ActorCriticSentBertCand, ActorCriticSentWordBertCand
from data.batcher import conver2id, pad_batch_tensorize, bert_tokenizer
from data.data import CnnDmDataset


try:
    DATASET_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')

class DecodeDataset(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split):
        #assert split in ['val', 'test']
        assert 'train' not in split
        super().__init__(split, DATASET_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        return art_sents


def make_html_safe(s):
    """Rouge use html, has to make output html safe"""
    return s.replace("<", "&lt;").replace(">", "&gt;")


def load_best_ckpt(model_dir, reverse=False):
    """ reverse=False->loss, reverse=True->reward/score"""
    ckpts = os.listdir(join(model_dir, 'ckpt'))
    ckpt_matcher = re.compile('^ckpt-.*-[0-9]*')
    ckpts = sorted([c for c in ckpts if ckpt_matcher.match(c)],
                   key=lambda c: float(c.split('-')[1]), reverse=reverse)
    print('loading checkpoint {}...'.format(ckpts[0]))
    ckpt = torch.load(join(model_dir, 'ckpt/{}'.format(ckpts[0])), map_location=lambda storage, loc: storage)['state_dict']
    print("torch.load")
    return ckpt


class Abstractor(object):
    def __init__(self, abs_dir, max_len=30, cuda=True):
        abs_meta = json.load(open(join(abs_dir, 'meta.json')))
        assert abs_meta['net'] == 'base_abstractor'
        abs_args = abs_meta['net_args']
        abs_ckpt = load_best_ckpt(abs_dir)
        word2id = pkl.load(open(join(abs_dir, 'vocab.pkl'), 'rb'))
        abstractor = CopySumm(**abs_args)
        abstractor.load_state_dict(abs_ckpt)
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = abstractor.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}
        self._max_len = max_len
        self.n_hidden = self._net.n_hidden

        # set require_grad to false to all parameters
        for param in self._net.parameters():
            param.requires_grad = False

    def _prepro(self, raw_article_sents):
        ext_word2id = dict(self._word2id)
        ext_id2word = dict(self._id2word)
        for raw_words in raw_article_sents:
            for w in raw_words:
                if not w in ext_word2id:
                    ext_word2id[w] = len(ext_word2id)
                    ext_id2word[len(ext_id2word)] = w
        articles = conver2id(UNK, self._word2id, raw_article_sents)  # a list of list of int
        art_lens = [len(art) for art in articles]
        article = pad_batch_tensorize(articles, PAD, cuda=False
                                     ).to(self._device)
        extend_arts = conver2id(UNK, ext_word2id, raw_article_sents)
        extend_art = pad_batch_tensorize(extend_arts, PAD, cuda=False
                                        ).to(self._device)
        extend_vsize = len(ext_word2id)
        dec_args = (article, art_lens, extend_art, extend_vsize,
                    START, END, UNK, self._max_len)
        return dec_args, ext_id2word

    def __call__(self, raw_article_sents):
        self._net.eval()
        dec_args, id2word = self._prepro(raw_article_sents)
        decs, attns = self._net.batch_decode(*dec_args)
        def argmax(arr, keys):
            return arr[max(range(len(arr)), key=lambda i: keys[i].item())]
        dec_sents = []  # a list of token list, len=batch_size
        # convert idx to words
        for i, raw_words in enumerate(raw_article_sents):  # each input sentence
            dec = []
            for id_, attn in zip(decs, attns):  # id_: a tensor with size = batch_size
                if id_[i] == END:
                    break
                elif id_[i] == UNK:  # replace unk word with highest attention word
                    dec.append(argmax(raw_words, attn[i]))
                else:
                    dec.append(id2word[id_[i].item()])
            dec_sents.append(dec)
        return dec_sents

    def encode(self, article_sents, sent_nums):
        # return the initial decode hidden state
        _, (init_dec_states, _) = self._net.encode(article_sents, sent_nums)
        return init_dec_states[0]  # [num_sents, abstractor_n_hidden]

class CompressionControlAbstractor(object):
    def __init__(self, abs_dir, max_len=30, cuda=True):
        abs_meta = json.load(open(join(abs_dir, 'meta.json')))
        assert abs_meta['net'] == 'controllable_abstractor'
        abs_args = abs_meta['net_args']
        abs_ckpt = load_best_ckpt(abs_dir)
        word2id = pkl.load(open(join(abs_dir, 'vocab.pkl'), 'rb'))
        abstractor = CompressControlSumm(**abs_args)
        abstractor.load_state_dict(abs_ckpt)
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = abstractor.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}
        self._max_len = max_len
        self.n_hidden = self._net.n_hidden

        # set require_grad to false to all parameters
        for param in self._net.parameters():
            param.requires_grad = False

    def _prepro(self, raw_article_sents, raw_compression_levels):
        ext_word2id = dict(self._word2id)
        ext_id2word = dict(self._id2word)
        for raw_words in raw_article_sents:
            for w in raw_words:
                if not w in ext_word2id:
                    ext_word2id[w] = len(ext_word2id)
                    ext_id2word[len(ext_id2word)] = w
        articles = conver2id(UNK, self._word2id, raw_article_sents)  # a list of list of int
        art_lens = [len(art) for art in articles]
        article = pad_batch_tensorize(articles, PAD, cuda=False
                                     ).to(self._device)
        compression_levels = torch.LongTensor(raw_compression_levels).to(self._device)
        extend_arts = conver2id(UNK, ext_word2id, raw_article_sents)
        extend_art = pad_batch_tensorize(extend_arts, PAD, cuda=False
                                        ).to(self._device)
        extend_vsize = len(ext_word2id)
        dec_args = (article, art_lens, compression_levels, extend_art, extend_vsize,
                    START, END, UNK, self._max_len)
        return dec_args, ext_id2word

    def __call__(self, raw_article_sents, raw_compression_levels):
        self._net.eval()
        dec_args, id2word = self._prepro(raw_article_sents, raw_compression_levels)
        decs, attns = self._net.batch_decode(*dec_args)
        def argmax(arr, keys):
            return arr[max(range(len(arr)), key=lambda i: keys[i].item())]
        dec_sents = []  # a list of token list, len=batch_size
        # convert idx to words
        for i, raw_words in enumerate(raw_article_sents):  # each input sentence
            dec = []
            for id_, attn in zip(decs, attns):  # id_: a tensor with size = batch_size
                if id_[i] == END:
                    break
                elif id_[i] == UNK:  # replace unk word with highest attention word
                    dec.append(argmax(raw_words, attn[i]))
                else:
                    dec.append(id2word[id_[i].item()])
            dec_sents.append(dec)
        return dec_sents


class ConditionalAbstractor(object):
    def __init__(self, abs_dir, max_len=30, cuda=True):
        abs_meta = json.load(open(join(abs_dir, 'meta.json')))
        assert abs_meta['net'] == 'conditional_abstractor'
        abs_args = abs_meta['net_args']
        abs_ckpt = load_best_ckpt(abs_dir)
        word2id = pkl.load(open(join(abs_dir, 'vocab.pkl'), 'rb'))

        abstractor = CopyCondSumm(**abs_args)

        abstractor.load_state_dict(abs_ckpt)
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = abstractor.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}
        self._max_len = max_len
        self.n_hidden = self._net.n_hidden

        # set require_grad to false to all parameters
        for param in self._net.parameters():
            param.requires_grad = False

    def _prepro(self, sequential_raw_article_sents):
        ext_word2id = dict(self._word2id)
        ext_id2word = dict(self._id2word)
        articles_sequential = []
        art_lens_sequential = []
        extend_arts_sequential = []

        for raw_article_sents in sequential_raw_article_sents:
            for raw_words in raw_article_sents:
                for w in raw_words:
                    if not w in ext_word2id:
                        ext_word2id[w] = len(ext_word2id)
                        ext_id2word[len(ext_id2word)] = w
            # convert article to idx and pad and append to list
            articles = conver2id(UNK, self._word2id, raw_article_sents)
            article = pad_batch_tensorize(articles, PAD, cuda=False
                                         ).to(self._device)
            articles_sequential.append(article)
            # compute art lens
            art_lens = [len(art) for art in articles]
            art_lens_sequential.append(art_lens)
            # convert article to idx with oov and pad and append to list
            extend_arts = conver2id(UNK, ext_word2id, raw_article_sents)
            extend_art = pad_batch_tensorize(extend_arts, PAD, cuda=False
                                            ).to(self._device)
            extend_arts_sequential.append(extend_art)

        # construct an initial memory sentence with only the <empty_mem> token
        # repeat it for every article
        # convert it to idx and pad
        init_mems = conver2id(UNK, self._word2id, [['<empty_mem>'] for _ in range(len(sequential_raw_article_sents[0]))])
        init_mem = pad_batch_tensorize(init_mems, PAD, cuda=False).to(self._device)
        init_mem_lens = [1] * len(sequential_raw_article_sents[0])

        assert init_mem.size() == torch.Size([len(sequential_raw_article_sents[0]), 1])

        extend_vsize = len(ext_word2id)
        dec_args = (articles_sequential, art_lens_sequential, init_mem, init_mem_lens, extend_arts_sequential, extend_vsize,
                    START, END, UNK, self._max_len)
        return dec_args, ext_id2word

    def __call__(self, sequential_raw_article_sents, sequential_article_ids, num_articles):
        # raw_article_sents: a list of token list
        self._net.eval()
        dec_args, id2word = self._prepro(sequential_raw_article_sents)
        # dec_args = (article, art_lens, extend_art, extend_vsize, START, END, UNK, self._max_len)
        articles_sequential, art_lens_sequential, external_memory, external_memory_lens, extend_arts_sequential, extend_vsize, start, end, unk, max_len = dec_args
        memory_sents_all = [[] for _ in range(num_articles)]  # len=batch_size
        dec_sents_raw_all = [[] for _ in range(num_articles)]  # len=batch_size

        for i, (articles_sents_i, art_lens_i, extend_arts_sents_i, raw_art_sents_i, article_ids) in enumerate(zip(articles_sequential, art_lens_sequential, extend_arts_sequential, sequential_raw_article_sents, sequential_article_ids)):

            if i > 0:
                # construct memory and memory_lens for the current step
                external_memory = pad_batch_tensorize([memory_sents_all[article_id] for article_id in article_ids], PAD,
                                                      cuda=False).to(self._device)
                external_memory_lens = [len(memory_sents_all[article_id]) for article_id in article_ids]

            # decode
            decs, attns = self._net.batch_decode(articles_sents_i, art_lens_i, external_memory, external_memory_lens, extend_arts_sents_i, extend_vsize, start, end, unk, max_len)
            # decs: a list of tensor with size [batch_size]
            # preprocess dec. dec_sents: a list of token list, len=batch_size
            dec_sents_raw, dec_sents = self._process_dec_out(decs, attns, raw_art_sents_i, id2word, self._word2id)

            # update memory_sents_all
            assert len(dec_sents) == len(article_ids)
            for dec_sent, article_id, dec_sent_raw in zip(dec_sents, article_ids, dec_sents_raw):
                memory_sents_all[article_id] += dec_sent
                dec_sents_raw_all[article_id].append(dec_sent_raw)

        return list(concat(dec_sents_raw_all))


    def _process_dec_out(self, decs, attns, raw_article_sents, id2word, word2id):
        def argmax(arr, keys):
            return arr[max(range(len(arr)), key=lambda i: keys[i].item())]
        dec_sents = []  # a list of token list, len=batch_size
        dec_sents_idx = []  # a list of int list, len=batch_size
        # convert idx to words
        for i, raw_words in enumerate(raw_article_sents):  # each input sentence
            dec = []
            dec_idx = []
            for id_, attn in zip(decs, attns):  # id_: a tensor with size = batch_size
                id_i = id_[i].item()
                if id_i == END:
                    break
                elif id_i == UNK:  # replace unk word with highest attention word
                    max_attn_src_word = argmax(raw_words, attn[i])
                    dec.append(max_attn_src_word)
                    #dec_idx.append(word2id[max_attn_src_word])
                else:
                    dec.append(id2word[id_i])
                dec_idx.append(id_i if id_i < len(word2id) else UNK)  # should not include oov, otherwise we cannot compute an embedding
            dec_sents.append(dec)
            dec_sents_idx.append(dec_idx)
        return dec_sents, dec_sents_idx


class BeamAbstractor(Abstractor):
    def __call__(self, raw_article_sents, beam_size=5, diverse=1.0):
        self._net.eval()
        dec_args, id2word = self._prepro(raw_article_sents)
        dec_args = (*dec_args, beam_size, diverse)
        all_beams = self._net.batched_beamsearch(*dec_args)
        all_beams = list(starmap(_process_beam(id2word),
                                 zip(all_beams, raw_article_sents)))
        return all_beams


class BeamCompressionControlAbstractor(CompressionControlAbstractor):
    def __call__(self, raw_article_sents, raw_compression_levels, beam_size=5, diverse=1.0):
        self._net.eval()
        dec_args, id2word = self._prepro(raw_article_sents, raw_compression_levels)
        dec_args = (*dec_args, beam_size, diverse)
        all_beams = self._net.batched_beamsearch(*dec_args)
        all_beams = list(starmap(_process_beam(id2word),
                                 zip(all_beams, raw_article_sents)))
        return all_beams


@curry
def _process_beam(id2word, beam, art_sent):
    def process_hyp(hyp):
        seq = []
        for i, attn in zip(hyp.sequence[1:], hyp.attns[:-1]):
            if i == UNK:
                copy_word = art_sent[max(range(len(art_sent)),
                                         key=lambda j: attn[j].item())]
                seq.append(copy_word)
            else:
                seq.append(id2word[i])
        hyp.sequence = seq
        del hyp.hists
        del hyp.attns
        return hyp
    return list(map(process_hyp, beam))


class BeamConditionalAbstractor(ConditionalAbstractor):
    def __call__(self, sequential_raw_article_sents, sequential_article_ids, num_articles, beam_size=5, diverse=1.0):
        self._net.eval()
        dec_args, id2word = self._prepro(sequential_raw_article_sents)
        # (articles_sequential, art_lens_sequential, init_mem, init_mem_lens, extend_arts_sequential, extend_vsize,
        #            START, END, UNK, self._max_len)
        # dec_args = (*dec_args, beam_size, diverse)
        articles_sequential, art_lens_sequential, external_memory, external_memory_lens, extend_arts_sequential, extend_vsize, start, end, unk, max_len = dec_args
        memory_sents_all = [[] for _ in range(num_articles)]  # len=batch_size
        dec_sents_raw_all = [[] for _ in range(num_articles)]  # len=batch_size

        for i, (articles_sents_i, art_lens_i, extend_arts_sents_i, raw_art_sents_i, article_ids) in enumerate(
            zip(articles_sequential, art_lens_sequential, extend_arts_sequential, sequential_raw_article_sents,
                sequential_article_ids)):
            if i > 0:
                # construct memory and memory_lens for the current step
                external_memory = pad_batch_tensorize([memory_sents_all[article_id] for article_id in article_ids],
                                                      PAD,
                                                      cuda=False).to(self._device)
                external_memory_lens = [len(memory_sents_all[article_id]) for article_id in article_ids]

            # beam search
            all_beams = self._net.batched_beamsearch(articles_sents_i, art_lens_i, external_memory, external_memory_lens, extend_arts_sents_i, extend_vsize, start, end, unk, max_len, beam_size, diverse)
            all_beams = list(starmap(_process_cond_beam(id2word, self._word2id), zip(all_beams, raw_art_sents_i)))
            all_beams = rerank_one(all_beams)

            assert len(all_beams) == len(article_ids)
            # all_beams: a list of list of hyp
            for beam, article_id in zip(all_beams, article_ids):
                memory_sents_all[article_id] += beam.sequence_idx
                dec_sents_raw_all[article_id].append(beam.sequence)
                #memory_sents_all[article_id] += beam[0].sequence_idx
                #dec_sents_raw_all[article_id].append(beam[0].sequence)

        return list(concat(dec_sents_raw_all))


def rerank_one(beams):
    @curry
    def process_beam(beam):
        for b in beam:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam
    beams = map(process_beam, beams)
    beams = [max(hyps, key=_compute_score) for hyps in beams]
    return beams


def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))


def _compute_score(hyp):
    repeat = sum(c-1 for g, c in hyp.gram_cnt.items() if c > 1)
    lp = hyp.logprob / len(hyp.sequence)
    return (-repeat, lp)


class MemroyBeamConditionalAbstractor(ConditionalAbstractor):
    def _prepro(self, raw_article_sents, raw_memory_sents):
        ext_word2id = dict(self._word2id)
        ext_id2word = dict(self._id2word)
        for raw_words in raw_article_sents:
            for w in raw_words:
                if not w in ext_word2id:
                    ext_word2id[w] = len(ext_word2id)
                    ext_id2word[len(ext_id2word)] = w
        articles = conver2id(UNK, self._word2id, raw_article_sents)  # a list of list of int
        art_lens = [len(art) for art in articles]
        article = pad_batch_tensorize(articles, PAD, cuda=False).to(self._device)

        memories = conver2id(UNK, self._word2id, raw_memory_sents)  # a list of list of int
        mem_lens = [len(mem) for mem in memories]
        memory = pad_batch_tensorize(memories, PAD, cuda=False).to(self._device)

        extend_arts = conver2id(UNK, ext_word2id, raw_article_sents)
        extend_art = pad_batch_tensorize(extend_arts, PAD, cuda=False
                                        ).to(self._device)

        extend_vsize = len(ext_word2id)
        dec_args = (article, art_lens, memory, mem_lens, extend_art, extend_vsize,
                    START, END, UNK, self._max_len)
        return dec_args, ext_id2word

    def __call__(self, raw_article_sents, raw_memory_sents, beam_size=5, diverse=1.0):
        self._net.eval()
        dec_args, id2word = self._prepro(raw_article_sents, raw_memory_sents)
        dec_args = (*dec_args, beam_size, diverse)
        all_beams = self._net.batched_beamsearch(*dec_args)
        all_beams = list(starmap(_process_beam(id2word), zip(all_beams, raw_article_sents)))
        return all_beams


@curry
def _process_cond_beam(id2word, word2id, beam, art_sent):
    def process_hyp(hyp):
        seq = []
        seq_idx = []
        for i, attn in zip(hyp.sequence[1:], hyp.attns[:-1]):
            if i == UNK:
                copy_word = art_sent[max(range(len(art_sent)),
                                         key=lambda j: attn[j].item())]
                seq.append(copy_word)
            else:
                seq.append(id2word[i])
            seq_idx.append(i if i < len(word2id) else UNK)
        hyp.sequence = seq
        hyp.sequence_idx = seq_idx
        del hyp.hists
        del hyp.attns
        return hyp
    return list(map(process_hyp, beam))


class Extractor(object):
    def __init__(self, ext_dir, max_ext=5, cuda=True, disable_selected_mask=False):
        ext_meta = json.load(open(join(ext_dir, 'meta.json')))
        if ext_meta['net'] == 'ml_ff_extractor':
            ext_cls = ExtractSumm
        elif ext_meta['net'] == 'ml_rnn_extractor':
            ext_cls = PtrExtractSumm
        elif ext_meta['net'] == 'ml_rewritten_rnn_extractor':
            ext_cls = PtrExtractRewrittenSumm
        elif ext_meta['net'] == 'ml_rewritten_bert_rnn_extractor':
            ext_cls = PtrExtractRewrittenBertSumm
        elif ext_meta['net'] == 'ml_rewritten_sent_bert_rnn_extractor':
            ext_cls = PtrExtractRewrittenSentBertSumm
        elif ext_meta['net'] == 'ml_rewritten_sent_word_bert_rnn_extractor':
            ext_cls = PtrExtractRewrittenSentWordBertSumm
        else:
            raise ValueError()
        ext_ckpt = load_best_ckpt(ext_dir)
        ext_args = ext_meta['net_args']
        extractor = ext_cls(**ext_args)
        extractor.load_state_dict(ext_ckpt)
        word2id = pkl.load(open(join(ext_dir, 'vocab.pkl'), 'rb'))
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = extractor.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}
        self._max_ext = max_ext
        self._disable_selected_mask = disable_selected_mask
        if "bert" in ext_meta['net']:
            self.emb_type = "bert"
        else:
            self.emb_type = "w2v"

    def __call__(self, raw_article_sents):
        self._net.eval()
        n_art = len(raw_article_sents)
        if self.emb_type == "w2v":
            articles = conver2id(UNK, self._word2id, raw_article_sents)
        else:
            articles = [bert_tokenizer.convert_tokens_to_ids(sentence) for sentence in raw_article_sents]
        article = pad_batch_tensorize(articles, PAD, cuda=False
                                     ).to(self._device)
        indices = self._net.extract([article], k=min(n_art, self._max_ext), disable_selected_mask=self._disable_selected_mask)
        return indices


class ArticleBatcher(object):
    def __init__(self, word2id, emb_type, cuda=True):
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._word2id = word2id
        self._device = torch.device('cuda' if cuda else 'cpu')
        self.emb_type = emb_type

    def __call__(self, raw_article_sents):
        if self.emb_type == "w2v":
            articles = conver2id(UNK, self._word2id, raw_article_sents)
        else:
            articles = [bert_tokenizer.convert_tokens_to_ids(sentence) for sentence in raw_article_sents]
        article = pad_batch_tensorize(articles, PAD, cuda=False
                                     ).to(self._device)
        return article

class RLExtractor(object):
    def __init__(self, ext_dir, cuda=True, num_candidates = 1, disable_selected_mask=False):
        ext_meta = json.load(open(join(ext_dir, 'meta.json')))
        #assert ext_meta['net'] in ['rnn-ext_abs_rl', 'rewritten_rnn-ext_abs_rl']
        ext_args = ext_meta['net_args']['extractor']['net_args']
        word2id = pkl.load(open(join(ext_dir, 'agent_vocab.pkl'), 'rb'))
        self.net_type = ext_meta['net']
        if "bert" in self.net_type:
            self.emb_type = "bert"
        else:
            self.emb_type = "w2v"
        if self.net_type == 'rnn-ext_abs_rl':
            extractor = PtrExtractSumm(**ext_args)
            assert num_candidates == 1, "When using PtrExtractSumm, num_candidates should be 1."
            agent = ActorCritic(extractor._sent_enc,
                                extractor._art_enc,
                                extractor._extractor,
                                ArticleBatcher(word2id, self.emb_type, cuda))
        elif self.net_type == 'rewritten_rnn-ext_abs_rl':
            extractor = PtrExtractRewrittenSumm(**ext_args)
            assert num_candidates > 1 and num_candidates == extractor.num_candidates
            agent = ActorCriticCand(extractor._candidate_sent_enc,
                            extractor._candidate_agg,
                            extractor._art_enc,
                            extractor._extractor,
                            ArticleBatcher(word2id, self.emb_type, cuda),
                            num_candidates)
        elif self.net_type == 'rewritten_bert_rnn-ext_abs_rl':
            raise ValueError
        elif self.net_type == 'rewritten_sent_bert_rnn-ext_abs_rl':
            extractor = PtrExtractRewrittenSentBertSumm(**ext_args)
            assert num_candidates > 1 and num_candidates == extractor.num_candidates
            agent = ActorCriticSentBertCand(extractor._candidate_sent_encode,
                                            extractor._bert_w,
                                            extractor._candidate_agg,
                                            extractor._art_enc,
                                            extractor._extractor,
                                            ArticleBatcher(word2id, self.emb_type, cuda),
                                            num_candidates)
        elif self.net_type == 'rewritten_sent_word_bert_rnn-ext_abs_rl':
            extractor = PtrExtractRewrittenSentWordBertSumm(**ext_args)
            print(extractor)
            assert num_candidates > 1 and num_candidates == extractor.num_candidates
            agent = ActorCriticSentWordBertCand(extractor._sentence_encoder,
                                            extractor._bert_w,
                                            extractor._candidate_sent_enc,
                                            extractor._candidate_agg,
                                            extractor._art_enc,
                                            extractor._extractor,
                                            ArticleBatcher(word2id, self.emb_type, cuda),
                                            num_candidates)
        else:
            raise ValueError
        ext_ckpt = load_best_ckpt(ext_dir, reverse=True)
        agent.load_state_dict(ext_ckpt)
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = agent.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}
        self._disable_selected_mask = disable_selected_mask

    def __call__(self, raw_article_sents):
        self._net.eval()
        indices = self._net(raw_article_sents, self._disable_selected_mask)
        return indices
