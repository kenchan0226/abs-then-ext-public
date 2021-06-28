""" train extractor (ML)"""
import argparse
import json
import os
from os.path import join, exists
import pickle as pkl

from cytoolz import compose

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model.extract import ExtractSumm, PtrExtractSumm, PtrExtractRewrittenSumm, PtrExtractRewrittenBertSumm, PtrExtractRewrittenSentBertSumm, PtrExtractRewrittenSentWordBertSumm
from model.util import sequence_loss
from training import get_basic_grad_fn, basic_validate
from training import BasicPipeline, BasicTrainer

from utils import PAD, UNK
from utils import make_vocab, make_embedding

from data.data import CnnDmDataset
from data.batcher import coll_fn_extract, prepro_fn_extract
from data.batcher import convert_batch_extract_ff, batchify_fn_extract_ff
from data.batcher import convert_batch_extract_ptr, batchify_fn_extract_ptr
from data.batcher import BucketedGenerater

import numpy as np
import random


BUCKET_SIZE = 6400
torch.backends.cudnn.enabled = False

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')

if "MODEL_CACHE" in os.environ:
    CACHE_DIR = os.environ['MODEL_CACHE']

class ExtractDataset(CnnDmDataset):
    """ article sentences -> extraction indices
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, extracts = js_data['article'], js_data['extracted']
        return art_sents, extracts

class ExtractDatasetStop(CnnDmDataset):
    """ article sentences -> extraction indices
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, extracts = js_data['article'], js_data['extracted']
        extracts.append(len(art_sents))
        return art_sents, extracts

def build_batchers(net_type, word2id, cuda, train_set_folder, valid_set_folder, debug, auto_stop=False, emb_type="w2v", num_candidates=1):
    assert net_type in ['ff', 'rnn', 'rewritten_rnn', 'rewritten_bert_rnn', 'rewritten_sent_bert_rnn', 'rewritten_sent_word_bert_rnn']
    prepro = prepro_fn_extract(args.max_word, args.max_sent, emb_type, num_candidates)
    def sort_key(sample):
        src_sents, _ = sample
        return len(src_sents)
    batchify_fn = (batchify_fn_extract_ff if net_type == 'ff'
                   else batchify_fn_extract_ptr)
    convert_batch = (convert_batch_extract_ff if net_type == 'ff'
                     else convert_batch_extract_ptr)
    batchify = compose(batchify_fn(PAD, cuda=cuda),
                       convert_batch(UNK, word2id, emb_type))

    if auto_stop:
        dataset_class = ExtractDatasetStop
    else:
        dataset_class = ExtractDataset

    train_loader = DataLoader(
        dataset_class(train_set_folder), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn_extract
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=not debug)

    val_loader = DataLoader(
        dataset_class(valid_set_folder), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn_extract
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                    single_run=True, fork=not debug)
    return train_batcher, val_batcher


def configure_net(net_type, vocab_size, emb_dim, conv_hidden,
                  lstm_hidden, lstm_layer, bidirectional, num_candidates=-1, candidate_agg_type='mean', auto_stop=False):
    assert net_type in ['ff', 'rnn', 'rewritten_rnn', 'rewritten_bert_rnn', 'rewritten_sent_bert_rnn', 'rewritten_sent_word_bert_rnn']
    net_args = {}
    net_args['vocab_size']    = vocab_size
    net_args['emb_dim']       = emb_dim
    net_args['conv_hidden']   = conv_hidden
    net_args['lstm_hidden']   = lstm_hidden
    net_args['lstm_layer']    = lstm_layer
    net_args['bidirectional'] = bidirectional
    net_args['auto_stop'] = auto_stop
    if "rewritten" in net_type:
        assert num_candidates > 0
        net_args['num_candidates'] = num_candidates
        net_args['candidate_agg_type'] = candidate_agg_type
        if "bert" in net_type:
            net_args["cache_dir"] = CACHE_DIR

    if net_type == 'ff':
        net = ExtractSumm(**net_args)
    elif net_type == 'rnn':
        net = PtrExtractSumm(**net_args)
    elif net_type == 'rewritten_rnn':
        net = PtrExtractRewrittenSumm(**net_args)
    elif net_type == 'rewritten_bert_rnn':
        net = PtrExtractRewrittenBertSumm(**net_args)
    elif net_type == 'rewritten_sent_bert_rnn':
        net = PtrExtractRewrittenSentBertSumm(**net_args)
    elif net_type == 'rewritten_sent_word_bert_rnn':
        net = PtrExtractRewrittenSentWordBertSumm(**net_args)
    else:
        raise ValueError
    # net = (ExtractSumm(**net_args) if net_type == 'ff' else PtrExtractSumm(**net_args))
    print(net)

    # print number of trainable parameters
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}".format(trainable_params))

    return net, net_args


def configure_training(net_type, opt, lr, clip_grad, lr_decay, batch_size):
    """ supports Adam optimizer only"""
    assert opt in ['adam']
    assert net_type in ['ff', 'rnn', 'rewritten_rnn', 'rewritten_bert_rnn', 'rewritten_sent_bert_rnn', 'rewritten_sent_word_bert_rnn']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay

    if net_type == 'ff':
        criterion = lambda logit, target: F.binary_cross_entropy_with_logits(
            logit, target, reduce=False)
    else:
        ce = lambda logit, target: F.cross_entropy(logit, target, reduce=False)
        def criterion(logits, targets):
            return sequence_loss(logits, targets, ce, pad_idx=-1)

    return criterion, train_params


def main(args):
    assert args.net_type in ['ff', 'rnn', 'rewritten_rnn', 'rewritten_bert_rnn', 'rewritten_sent_bert_rnn', 'rewritten_sent_word_bert_rnn']
    if "bert" in args.net_type:
        args.emb_type = "bert"
    else:
        args.emb_type = "w2v"
    # create data batcher, vocabulary
    # batcher
    with open(join(DATA_DIR, 'vocab_cnt.pkl'), 'rb') as f:
        wc = pkl.load(f)
    word2id = make_vocab(wc, args.vsize)
    train_batcher, val_batcher = build_batchers(args.net_type, word2id,
                                                args.cuda, args.train_set_folder, args.valid_set_folder, args.debug, args.auto_stop, args.emb_type, args.num_candidates)

    # make net
    net, net_args = configure_net(args.net_type,
                                  len(word2id), args.emb_dim, args.conv_hidden,
                                  args.lstm_hidden, args.lstm_layer, args.bi, args.num_candidates, args.candidate_agg_type, args.auto_stop)
    if args.w2v and "bert" not in args.net_type:
        # NOTE: the pretrained embedding having the same dimension
        #       as args.emb_dim should already be trained
        embedding, _ = make_embedding(
            {i: w for w, i in word2id.items()}, args.w2v)
        net.set_embedding(embedding)

    # configure training setting
    criterion, train_params = configure_training(
        args.net_type, 'adam', args.lr, args.clip, args.decay, args.batch
    )

    # save experiment setting
    if not exists(args.path):
        os.makedirs(args.path)
    with open(join(args.path, 'vocab.pkl'), 'wb') as f:
        pkl.dump(word2id, f, protocol=4)
    meta = {}
    meta['net']           = 'ml_{}_extractor'.format(args.net_type)
    meta['net_args']      = net_args
    meta['traing_params'] = train_params
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    # prepare trainer
    val_fn = basic_validate(net, criterion)
    grad_fn = get_basic_grad_fn(net, args.clip)
    optimizer = optim.Adam(net.parameters(), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=args.decay, min_lr=args.min_lr,
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
        description='training of the feed-forward extractor (ff-ext, ML)'
    )
    parser.add_argument('--path', required=True, help='root of the model')

    # model options
    parser.add_argument('--net-type', action='store', default='rnn',
                        help='model type of the extractor (ff/rnn, rewritten_rnn)')
    parser.add_argument('--vsize', type=int, action='store', default=30000,
                        help='vocabulary size')
    parser.add_argument('--emb_dim', type=int, action='store', default=128,
                        help='the dimension of word embedding')
    parser.add_argument('--w2v', action='store',
                        help='use pretrained word2vec embedding')
    parser.add_argument('--conv_hidden', type=int, action='store', default=100,
                        help='the number of hidden units of Conv')
    parser.add_argument('--lstm_hidden', type=int, action='store', default=256,
                        help='the number of hidden units of lSTM')
    parser.add_argument('--lstm_layer', type=int, action='store', default=1,
                        help='the number of layers of LSTM Encoder')
    parser.add_argument('--no-bi', action='store_true',
                        help='disable bidirectional LSTM encoder')

    # length limit
    parser.add_argument('--max_word', type=int, action='store', default=100,
                        help='maximun words in a single article sentence')
    parser.add_argument('--max_sent', type=int, action='store', default=60,
                        help='maximun sentences in an article article')
    # training options
    parser.add_argument('--lr', type=float, action='store', default=1e-3,
                        help='learning rate')
    parser.add_argument('--min_lr', type=float, action='store', default=0,
                        help='minimum learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=0,
                        help='patience for learning rate decay')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='the training batch size')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=3000,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=5,
                        help='patience for early stopping')
    # args for rewritten_ptr_extractor
    parser.add_argument('--num_candidates', type=int, action='store', default=1,
                        help='number of candidates for each sentence')
    parser.add_argument('--auto_stop', action='store_true',
                        help='stop it when it points to the stop representation')

    parser.add_argument('--debug', action='store_true',
                        help='run in debugging mode')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--seed', type=int, action='store', default=9527,
                        help='disable GPU training')

    parser.add_argument('--train_set_folder', type=str, action='store', default="train",
                        help='The name of training set folder')
    parser.add_argument('--valid_set_folder', type=str, action='store', default="val",
                        help='The name of validation set folder')

    parser.add_argument('--candidate_agg_type', type=str, action='store', default="mean", choices=['mean', 'fc'],
                        help='The type of aggregator for candidate')

    args = parser.parse_args()
    args.bi = not args.no_bi
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    #args.max_sent = args.max_sent * args.num_candidates

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    main(args)