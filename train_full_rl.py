""" full training (train rnn-ext + abs + RL) """
import argparse
import json
import pickle as pkl
import os
from os.path import join, exists
from itertools import cycle
import sys

from toolz.sandbox.core import unzip
from cytoolz import identity, concat

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from utils import count_data
import pickle

from data.data import CnnDmDataset
from data.batcher import tokenize

from model.rl import ActorCritic, ActorCriticCand, ActorCriticSentBertCand, ActorCriticSentWordBertCand
from model.extract import PtrExtractSumm, PtrExtractRewrittenSumm, PtrExtractRewrittenBertSumm, PtrExtractRewrittenSentBertSumm, PtrExtractRewrittenSentWordBertSumm

from training import BasicTrainer
from rl import get_grad_fn
from rl import A2CPipeline
from decoding import load_best_ckpt
from decoding import Abstractor, ArticleBatcher, ConditionalAbstractor
from metric import compute_rouge_l, compute_rouge_l_summ, compute_rouge_n, compute_weighted_rouge_1_2
import random
import numpy as np


MAX_ABS_LEN = 30

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


class RLDataset(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        abs_sents = js_data['abstract']
        return art_sents, abs_sents


class RLDataset_backup(Dataset):
    def __init__(self, split):
        split_dir = os.path.join(DATA_DIR, split)
        cached_features_file = os.path.join(DATA_DIR, 'cached_' + split)

        if os.path.exists(cached_features_file):
            print("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            print("Creating features from dataset file at %s", DATA_DIR)
            self.examples = []
            n_data = count_data(split_dir)
            for i in range(n_data):
                js = json.load(open(join(split_dir, '{}.json'.format(i))))
                if js['article'] and js['abstract']:
                    doc_sent_list = js['article']
                    summary_sent_list = js['abstract']
                    self.examples.append( (doc_sent_list, summary_sent_list) )

            print("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=4)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


def load_ext_net(ext_dir, ext_type):
    ext_meta = json.load(open(join(ext_dir, 'meta.json')))
    #assert ext_meta['net'] == 'ml_rnn_extractor'
    assert 'ml_{}_extractor'.format(ext_type) == ext_meta['net']
    ext_ckpt = load_best_ckpt(ext_dir)
    print("finish load chkpt")
    #if '_extractor._stop' not in ext_ckpt:
    #    ext_ckpt['_extractor._stop'] = torch.zeros_like(ext_ckpt['_extractor._init_i'])
    ext_args = ext_meta['net_args']
    vocab = pkl.load(open(join(ext_dir, 'vocab.pkl'), 'rb'))
    print("finish load vocab")
    if ext_type == "rewritten_rnn":
        ext = PtrExtractRewrittenSumm(**ext_args)
    elif ext_type == "rewritten_bert_rnn":
        ext = PtrExtractRewrittenBertSumm(**ext_args)
    elif ext_type == "rewritten_sent_bert_rnn":
        ext = PtrExtractRewrittenSentBertSumm(**ext_args)
    elif ext_type == "rewritten_sent_word_bert_rnn":
        ext = PtrExtractRewrittenSentWordBertSumm(**ext_args)
    else:
        ext = PtrExtractSumm(**ext_args)
    ext.load_state_dict(ext_ckpt)
    print("loaded extractor")
    return ext, vocab


def configure_net(abs_dir, ext_dir, ext_type, emb_type, cuda, num_candidates=1, is_conditional_abs=False):
    """ load pretrained sub-modules and build the actor-critic network"""
    # load pretrained abstractor model
    """
    if abs_dir is not None:
        abstractor = Abstractor(abs_dir, MAX_ABS_LEN, cuda)
    else:
        abstractor = identity
    """
    if abs_dir is None or "rewritten" in ext_type:
        abstractor = identity
    elif is_conditional_abs:
        abstractor = ConditionalAbstractor(abs_dir, MAX_ABS_LEN, cuda)
    else:
        abstractor = Abstractor(abs_dir, MAX_ABS_LEN, cuda)

    # load ML trained extractor net and buiild RL agent
    extractor, agent_vocab = load_ext_net(ext_dir, ext_type)

    if ext_type == "rewritten_rnn":
        assert num_candidates == extractor.num_candidates
        agent = ActorCriticCand(extractor._candidate_sent_enc,
                        extractor._candidate_agg,
                        extractor._art_enc,
                        extractor._extractor,
                        ArticleBatcher(agent_vocab, emb_type, cuda),
                                num_candidates)
    elif ext_type == "rewritten_bert_rnn":
        raise ValueError
    elif ext_type == "rewritten_sent_bert_rnn":
        assert num_candidates == extractor.num_candidates
        agent = ActorCriticSentBertCand(extractor._sentence_encoder,
                                extractor._bert_w,
                                extractor._candidate_agg,
                                extractor._art_enc,
                                extractor._extractor,
                                ArticleBatcher(agent_vocab, emb_type, cuda),
                                num_candidates)
    elif ext_type == "rewritten_sent_word_bert_rnn":
        assert num_candidates == extractor.num_candidates
        agent = ActorCriticSentWordBertCand(extractor._sentence_encoder,
                                extractor._bert_w,
                                extractor._candidate_sent_enc,
                                extractor._candidate_agg,
                                extractor._art_enc,
                                extractor._extractor,
                                ArticleBatcher(agent_vocab, emb_type, cuda),
                                num_candidates)
    else:
        agent = ActorCritic(extractor._sent_enc,
                        extractor._art_enc,
                        extractor._extractor,
                        ArticleBatcher(agent_vocab, emb_type, cuda))
    if cuda:
        agent = agent.cuda()

    net_args = {}
    net_args['abstractor'] = (None if abs_dir is None
                              else json.load(open(join(abs_dir, 'meta.json'))))
    net_args['extractor'] = json.load(open(join(ext_dir, 'meta.json')))

    return agent, agent_vocab, abstractor, net_args


def configure_training(opt, lr, clip_grad, lr_decay, batch_size,
                       gamma, reward, stop_coeff, stop_reward):
    assert opt in ['adam']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay
    train_params['gamma']          = gamma
    train_params['reward']         = reward
    train_params['stop_coeff']     = stop_coeff
    train_params['stop_reward']    = stop_reward

    return train_params

def build_batchers(batch_size, train_set_folder, valid_set_folder, emb_type, num_candidates, max_word):
    def coll(batch):
        art_batch, abs_batch = unzip(batch)
        """
        # debug
        art_batch = list(art_batch)
        print("raw batch:")
        print(list(art_batch)[0][0])
        print(list(art_batch)[0][1])
        print(list(art_batch)[0][2])
        print(list(art_batch)[0][3])
        """
        #art_sents = list(filter(bool, map(tokenize(max_word, emb_type, num_candidates), art_batch)))
        #abs_sents = list(filter(bool, map(tokenize(max_word, emb_type, num_candidates), abs_batch)))

        art_sents = []
        #abs_sents = []
        raw_art_sents = []
        raw_abs_sents = []
        for art, abs in zip(art_batch, abs_batch):
            tokenized_art = tokenize(max_word, emb_type, num_candidates, art)[:args.max_sent]
            # tokenized_abs = tokenize(max_word, emb_type, num_candidates, abs)
            raw_art = [sent.split(" ")[:max_word] for sent in art]
            raw_abs = [sent.split(" ")[:max_word] for sent in abs]
            if tokenized_art and raw_abs:
                art_sents.append(tokenized_art)
                #abs_sents.append(tokenized_abs)
                raw_art_sents.append(raw_art)
                raw_abs_sents.append(raw_abs)

        return art_sents, raw_art_sents, raw_abs_sents
    loader = DataLoader(
        RLDataset(train_set_folder), batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
        collate_fn=coll
    )
    val_loader = DataLoader(
        RLDataset(valid_set_folder), batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
        collate_fn=coll
    )
    return cycle(loader), val_loader


def weighted_sum_rouge_12l(prediction_sent_list, abstract_sent_list):
    concated_prediction = list(concat(prediction_sent_list))
    concated_abstract = list(concat(abstract_sent_list))
    rouge_1 = compute_rouge_n(concated_prediction, concated_abstract, n=1, mode='f')
    rouge_2 = compute_rouge_n(concated_prediction, concated_abstract, n=2, mode='f')
    rouge_l = compute_rouge_l_summ(prediction_sent_list, abstract_sent_list, mode='f')
    weight = 1.0/3
    return weight * rouge_1 + weight * rouge_2 + weight * rouge_l


def train(args):
    if not exists(args.path):
        os.makedirs(args.path)

    if "bert" in args.ext_type:
        args.emb_type = "bert"
    else:
        args.emb_type = "w2v"

    print("emb_type")
    print(args.emb_type)

    # make net
    agent, agent_vocab, abstractor, net_args = configure_net(
        args.abs_dir, args.ext_dir, args.ext_type, args.emb_type, args.cuda, args.num_candidates, args.is_conditional_abs)

    print("network config")

    # configure training setting
    assert args.stop > 0
    train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch,
        args.gamma, args.reward, args.stop, 'rouge-1'
    )
    print("train config")
    train_batcher, val_batcher = build_batchers(args.batch, args.train_set_folder, args.valid_set_folder, args.emb_type, args.num_candidates, args.max_word)
    print("build batcher")
    if args.reward_type == 0:
        reward_fn = compute_rouge_l
        stop_reward_fn = compute_rouge_n(n=1)
    elif args.reward_type == 1 or args.reward_type == 2:
        reward_fn = compute_rouge_l_summ
        stop_reward_fn = compute_rouge_n(n=1)
    elif args.reward_type == 3:
        reward_fn = compute_rouge_l_summ(mode='r')
        stop_reward_fn = compute_rouge_n(n=1)
    elif args.reward_type == 4:
        reward_fn = compute_rouge_l_summ
        stop_reward_fn = compute_weighted_rouge_1_2(rouge_1_weight=0.5, mode='f')
    else:
        raise ValueError

    # save abstractor binary
    if args.abs_dir is not None:
        abs_ckpt = {}
        abs_ckpt['state_dict'] = load_best_ckpt(args.abs_dir)
        abs_vocab = pkl.load(open(join(args.abs_dir, 'vocab.pkl'), 'rb'))
        abs_dir = join(args.path, 'abstractor')
        os.makedirs(join(abs_dir, 'ckpt'))
        with open(join(abs_dir, 'meta.json'), 'w') as f:
            json.dump(net_args['abstractor'], f, indent=4)
        torch.save(abs_ckpt, join(abs_dir, 'ckpt/ckpt-0-0'))
        with open(join(abs_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(abs_vocab, f, protocol=4)
    # save configuration
    meta = {}
    meta['net']           = '{}-ext_abs_rl'.format(args.ext_type)  # 'ml_{}_extractor'
    meta['net_args']      = net_args
    meta['train_params']  = train_params
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)
    with open(join(args.path, 'agent_vocab.pkl'), 'wb') as f:
        pkl.dump(agent_vocab, f, protocol=4)

    # prepare trainer
    grad_fn = get_grad_fn(agent, args.clip)
    optimizer = optim.Adam(agent.parameters(), **train_params['optimizer'][1])
    if args.no_lr_decay:
        scheduler = None
    else:
        scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True,
                                      factor=args.decay, min_lr=args.min_lr,
                                      patience=args.lr_p)

    pipeline = A2CPipeline(meta['net'], agent, abstractor,
                           train_batcher, val_batcher,
                           optimizer, grad_fn,
                           reward_fn, args.gamma,
                           stop_reward_fn, args.stop, args.reward_type, args.disable_selected_mask, args.is_conditional_abs, args.debug)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler,
                           val_mode='score')

    print('start training with the following hyper-parameters:')
    print(meta)
    sys.stdout.flush()
    trainer.train()


if __name__ == '__main__':
    print("Go")
    parser = argparse.ArgumentParser(
        description='program to demo a Seq2Seq model'
    )
    parser.add_argument('--path', required=True, help='root of the model')


    # model options
    parser.add_argument('--abs_dir', action='store',
                        help='pretrained summarizer model root path')
    parser.add_argument('--ext_dir', action='store',
                        help='root of the extractor model')
    parser.add_argument('--ckpt', type=int, action='store', default=None,
                        help='ckeckpoint used decode')

    # training options
    parser.add_argument('--reward', action='store', default='rouge-l',
                        help='reward function for RL')
    parser.add_argument('--lr', type=float, action='store', default=1e-4,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=0,
                        help='patience for learning rate decay')
    parser.add_argument('--gamma', type=float, action='store', default=0.95,
                        help='discount factor of RL')
    parser.add_argument('--stop', type=float, action='store', default=1.0,
                        help='stop coefficient for rouge-1')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='the training batch size')
    parser.add_argument('--max_word', type=int, action='store', default=100,
                        help='maximun words in a single article sentence')
    parser.add_argument('--max_sent', type=int, action='store', default=60,
                        help='maximun sentences in an article article')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=1000,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=3,
                        help='patience for early stopping')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--disable_selected_mask', action='store_true',
                        help='disable the selection mask in the ptr network')

    parser.add_argument('--num_candidates', type=int, action='store', default=1,
                        help='number of candidates for each sentence.')
    parser.add_argument('--ext_type', type=str, action='store', default="rnn", choices=["rnn", "rewritten_rnn", "rewritten_bert_rnn", "rewritten_sent_bert_rnn", "rewritten_sent_word_bert_rnn"],
                        help='model type of the extractor (rnn, rewritten_rnn).')

    parser.add_argument('--train_set_folder', type=str, action='store', default="train",
                        help='The name of training set folder')
    parser.add_argument('--valid_set_folder', type=str, action='store', default="val",
                        help='The name of validation set folder')
    parser.add_argument('--debug', action='store_true',
                        help='use debug mode')

    parser.add_argument('--no_lr_decay', action='store_true',
                        help='do not use lr decay')
    parser.add_argument('--seed', type=int, action='store', default=9527,
                        help='disable GPU training')

    parser.add_argument('--min_lr', type=float, action='store', default=0,
                        help='minimum learning rate')
    parser.add_argument('--is_conditional_abs', action='store_true',
                        help='use conditional abstractor')

    parser.add_argument('--reward_type', type=int, action='store', default=0, choices=[0, 1, 2, 3, 4],
                        help='0: fast-rl, 1: weighted sum of Rouge 1, 2 and L + reward shaping')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    print("cuda is available:")
    print(args.cuda)
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    train(args)
