""" train the abstractor"""
import argparse
import json
import os
from os.path import join, exists
import pickle as pkl
import numpy as np
import random

from cytoolz import compose

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model.copy_summ import CopySumm
from model.util import sequence_loss
from training import get_basic_grad_fn, basic_validate
from training import BasicPipeline, BasicTrainer

from data.data import CnnDmDataset
from data.batcher import coll_fn, prepro_fn
from data.batcher import convert_batch_copy, batchify_fn_copy
from data.batcher import BucketedGenerater

from utils import PAD, UNK, START, END
from utils import make_vocab, make_embedding

# NOTE: bucket size too large may sacrifice randomness,
#       to low may increase # of PAD tokens
BUCKET_SIZE = 6400

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')

"""
class MatchDataset(CnnDmDataset):
    # single article sentence -> single abstract sentence
    # (dataset created by greedily matching ROUGE)
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, abs_sents, extracts = (
            js_data['article'], js_data['abstract'], js_data['extracted'])
        matched_arts = [art_sents[i] for i in extracts]
        return matched_arts, abs_sents[:len(extracts)]
"""

class MatchDataset(CnnDmDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split, filter_out_low_recall=False):
        super().__init__(split, DATA_DIR)
        self.filter_out_low_recall = filter_out_low_recall

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, abs_sents, extracts, scores = (
            js_data['article'], js_data['abstract'], js_data['extracted'], js_data['score'])

        if self.filter_out_low_recall:
            matched_arts = []
            matched_abss = []

            for ext_i, (ext, abst) in enumerate(zip(extracts, abs_sents)):
                if scores[ext_i] >= 0.2:
                    matched_arts.append(art_sents[ext])
                    matched_abss.append(abst)
        else:
            matched_arts = [art_sents[i] for i in extracts]
            matched_abss = abs_sents[:len(extracts)]

        #print("matched_arts: ")
        #print(matched_arts)
        #print("matched_abss: ")
        #print(matched_abss)

        return matched_arts, matched_abss


def configure_net(vocab_size, emb_dim,
                  n_hidden, bidirectional, n_layer):
    net_args = {}
    net_args['vocab_size']    = vocab_size
    net_args['emb_dim']       = emb_dim
    net_args['n_hidden']      = n_hidden
    net_args['bidirectional'] = bidirectional
    net_args['n_layer']       = n_layer

    net = CopySumm(**net_args)

    # print number of trainable parameters
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}".format(trainable_params))

    return net, net_args


def configure_training(opt, lr, clip_grad, lr_decay, batch_size):
    """ supports Adam optimizer only"""
    assert opt in ['adam']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay

    nll = lambda logit, target: F.nll_loss(logit, target, reduce=False)
    def criterion(logits, targets):
        return sequence_loss(logits, targets, nll, pad_idx=PAD)

    return criterion, train_params

def build_batchers(word2id, cuda, filter_out_low_recall, debug):
    prepro = prepro_fn(args.max_art, args.max_abs)
    def sort_key(sample):
        src, target = sample
        return (len(target), len(src))
    batchify = compose(
        batchify_fn_copy(PAD, START, END, cuda=cuda),  # padding
        convert_batch_copy(UNK, word2id)  # convert to idx
    )

    train_loader = DataLoader(
        MatchDataset('train', filter_out_low_recall), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=not debug)

    val_loader = DataLoader(
        MatchDataset('val', filter_out_low_recall), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                    single_run=True, fork=not debug)
    return train_batcher, val_batcher

def main(args):
    # create data batcher, vocabulary
    # batcher
    with open(join(DATA_DIR, 'vocab_cnt.pkl'), 'rb') as f:
        wc = pkl.load(f)
    word2id = make_vocab(wc, args.vsize)
    train_batcher, val_batcher = build_batchers(word2id,
                                                args.cuda, args.filter_out_low_recall, args.debug)

    # make net
    net, net_args = configure_net(len(word2id), args.emb_dim,
                                  args.n_hidden, args.bi, args.n_layer)
    if args.w2v:
        # NOTE: the pretrained embedding having the same dimension
        #       as args.emb_dim should already be trained
        embedding, _ = make_embedding(
            {i: w for w, i in word2id.items()}, args.w2v)
        net.set_embedding(embedding)

    # configure training setting
    criterion, train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch
    )

    # save experiment setting
    if not exists(args.path):
        os.makedirs(args.path)
    with open(join(args.path, 'vocab.pkl'), 'wb') as f:
        pkl.dump(word2id, f, pkl.HIGHEST_PROTOCOL)
    meta = {}
    meta['net'] = 'base_abstractor'
    meta['net_args']      = net_args
    meta['traing_params'] = train_params
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    # prepare trainer
    val_fn = basic_validate(net, criterion)
    grad_fn = get_basic_grad_fn(net, args.clip)
    optimizer = optim.Adam(net.parameters(), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)

    if args.cuda:
        net = net.cuda()
    pipeline = BasicPipeline(meta['net'], net,
                             train_batcher, val_batcher, args.batch, val_fn,
                             criterion, optimizer, grad_fn)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler)

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of the abstractor (ML)'
    )
    parser.add_argument('--path', required=True, help='root of the model')


    parser.add_argument('--vsize', type=int, action='store', default=30000,
                        help='vocabulary size')
    parser.add_argument('--emb_dim', type=int, action='store', default=128,
                        help='the dimension of word embedding')
    parser.add_argument('--w2v', action='store',
                        help='use pretrained word2vec embedding')
    parser.add_argument('--n_hidden', type=int, action='store', default=256,
                        help='the number of hidden units of LSTM')
    parser.add_argument('--n_layer', type=int, action='store', default=1,
                        help='the number of layers of LSTM')
    parser.add_argument('--no-bi', action='store_true',
                        help='disable bidirectional LSTM encoder')

    # length limit
    parser.add_argument('--max_art', type=int, action='store', default=100,
                        help='maximun words in a single article sentence')
    parser.add_argument('--max_abs', type=int, action='store', default=30,
                        help='maximun words in a single abstract sentence')
    # training options
    parser.add_argument('--lr', type=float, action='store', default=1e-3,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=0,
                        help='patience for learning rate decay')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='the training batch size')
    parser.add_argument('--seed', type=int, action='store', default=9527,
                        help='disable GPU training')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=3000,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=5,
                        help='patience for early stopping')

    parser.add_argument('--filter_out_low_recall', action='store_true',
                        help='Ignore the samples with low rouge-l recall')

    parser.add_argument('--debug', action='store_true',
                        help='run in debugging mode')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.bi = not args.no_bi
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    main(args)
