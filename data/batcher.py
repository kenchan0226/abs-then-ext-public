""" batching """
import random
from collections import defaultdict
import os
from toolz.sandbox import unzip
from cytoolz import curry, concat, compose
from cytoolz import curried

import torch
import torch.multiprocessing as mp
from transformers import BertTokenizer

CLS_WORD = '[CLS]'
SEP_WORD = '[SEP]'

if "MODEL_CACHE" in os.environ:
    CACHE_DIR = os.environ['MODEL_CACHE']
    bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", cache_dir=CACHE_DIR)
    BERT_MAX_ARTICLE_LEN = 512

# Batching functions
def coll_fn_control_gen(data):
    source_lists, target_lists, compression_level_lists = unzip(data)
    source_list = concat(source_lists)
    target_list = concat(target_lists)
    compression_level_list = concat(compression_level_lists)
    sources = []
    targets = []
    compression_levels = []
    for source_sent, target_sent, compression_level in zip(source_list, target_list, compression_level_list):
        if source_sent and target_sent:
            sources.append(source_sent)
            targets.append(target_sent)
            compression_levels.append(compression_level)
    return sources, targets, compression_levels


def coll_fn_cond_gen(data):
    source_lists, target_lists, memory_lists = unzip(data)
    source_list = concat(source_lists)
    target_list = concat(target_lists)
    memory_list = concat(memory_lists)
    sources = []
    targets = []
    memories = []
    for source_sent, target_sent, memory in zip(source_list, target_list, memory_list):
        if source_sent and target_sent:
            sources.append(source_sent)
            targets.append(target_sent)
            memories.append(memory)
    assert all(sources) and all(targets)
    return sources, targets, memories

def coll_fn(data):
    source_lists, target_lists = unzip(data)
    # NOTE: independent filtering works because
    #       source and targets are matched properly by the Dataset
    sources = list(filter(bool, concat(source_lists)))
    targets = list(filter(bool, concat(target_lists)))
    assert all(sources) and all(targets)
    return sources, targets

def coll_fn_extract(data):
    def is_good_data(d):
        """ make sure data is not empty"""
        source_sents, extracts = d
        return source_sents and extracts
    batch = list(filter(is_good_data, data))
    assert all(map(is_good_data, batch))
    return batch

@curry
def tokenize(max_len, emb_type, num_candidates, texts):
    if emb_type == "w2v":
        return [t.lower().split()[:max_len] for t in texts]
    else:
        #print("bert tokenize")
        return [[CLS_WORD] + bert_tokenizer.tokenize(t)[:max_len] + [SEP_WORD] for t in texts]
        """
        truncated_article = []
        #left = BERT_MAX_ARTICLE_LEN - 2
        left = [BERT_MAX_ARTICLE_LEN - 2] * num_candidates
        full_flag = False
        for sent_i in range(len(texts)//num_candidates):
            sent_cand_tokens = []
            #print("sent_i")
            #print(sent_i)
            for cand_i in range(num_candidates):
                #print("cand_i")
                #print(cand_i)
                cand_str = texts[sent_i * num_candidates + cand_i]
                cand_tokens = bert_tokenizer.tokenize(cand_str)
                if left[cand_i] >= len(cand_tokens):
                    sent_cand_tokens.append(cand_tokens)
                    left[cand_i] -= len(cand_tokens)
                else:
                    full_flag = True
                    break
                #print("sent_cand_tokens")
                #print(sent_cand_tokens)
                #print(left)
            if full_flag:
                break
            else:
                truncated_article += sent_cand_tokens
                #print("truncated_article")
                #print(truncated_article)
        """
        """
        for sent_i, sentence in enumerate(texts):
            tokens = bert_tokenizer.tokenize(sentence)
            tokens = tokens[:max_len]
            cand_i = sent_i % num_candidates
            if left[cand_i] >= len(tokens):
                truncated_article.append(tokens)
                left[cand_i] -= len(tokens)
            else:
                break
        #print("truncated article")
        #print(truncated_article)
        if len(truncated_article) % num_candidates != 0:
            print("left:")
            print(left)
            print("texts:")
            for sent in texts:
                print(sent)
            raise ValueError
        """
        """
        if len(truncated_article) == 0:
            print("texts")
            print(texts)
            print("len(texts)")
            print(len(texts))
            exit()
        """
        #return truncated_article

def conver2id(unk, word2id, words_list):
    word2id = defaultdict(lambda: unk, word2id)
    return [[word2id[w] for w in words] for words in words_list]

@curry
def prepro_fn(max_src_len, max_tgt_len, batch):
    sources, targets = batch
    sources = tokenize(max_src_len, "w2v", 1, sources)
    targets = tokenize(max_tgt_len, "w2v", 1, targets)
    batch = list(zip(sources, targets))
    return batch

@curry
def prepro_fn_cond(max_src_len, max_tgt_len, batch):
    sources, targets, memories = batch
    sources = tokenize(max_src_len, "w2v", 1, sources)
    targets = tokenize(max_tgt_len, "w2v", 1, targets)
    memories = tokenize(max_src_len * 4, memories)
    batch = list(zip(sources, targets, memories))
    return batch

@curry
def prepro_fn_control(max_src_len, max_tgt_len, batch):
    sources, targets, levels = batch
    sources = tokenize(max_src_len, "w2v", 1, sources)
    targets = tokenize(max_tgt_len, "w2v", 1, targets)
    batch = list(zip(sources, targets, levels))
    return batch

@curry
def prepro_fn_extract(max_src_len, max_src_num, emb_type, num_candidates, batch):
    def prepro_one(sample):
        source_sents, extracts = sample
        #if len(source_sents) % 2 != 0:
        #    print("source not 2")
        tokenized_sents = tokenize(max_src_len, emb_type, num_candidates, source_sents)[:max_src_num]
        #if len(tokenized_sents) % 2 != 0:
        #    print("tokenize not 2")
        #    print(len(source_sents))
        #    print(len(tokenized_sents))
        cleaned_extracts = list(filter(lambda e: e < len(tokenized_sents),
                                       extracts))
        return tokenized_sents, cleaned_extracts
    batch_prepro = []
    for sample in batch:
        tokenized_sents, cleaned_extracts = prepro_one(sample)
        if len(tokenized_sents) > 0:
            batch_prepro.append((tokenized_sents, cleaned_extracts))
    return batch_prepro
    #batch = list(map(prepro_one, batch))
    #return batch

@curry
def convert_batch(unk, word2id, batch):
    sources, targets = unzip(batch)
    sources = conver2id(unk, word2id, sources)
    targets = conver2id(unk, word2id, targets)
    batch = list(zip(sources, targets))
    return batch

@curry
def convert_batch_copy(unk, word2id, batch):
    sources, targets = map(list, unzip(batch))
    ext_word2id = dict(word2id)
    for source in sources:
        for word in source:
            if word not in ext_word2id:
                ext_word2id[word] = len(ext_word2id)
    src_exts = conver2id(unk, ext_word2id, sources)
    sources = conver2id(unk, word2id, sources)
    tar_ins = conver2id(unk, word2id, targets)
    targets = conver2id(unk, ext_word2id, targets)
    batch = list(zip(sources, src_exts, tar_ins, targets))
    return batch

@curry
def convert_batch_copy_cond(unk, word2id, batch):
    sources, targets, memories = map(list, unzip(batch))
    ext_word2id = dict(word2id)
    for source in sources:
        for word in source:
            if word not in ext_word2id:
                ext_word2id[word] = len(ext_word2id)
    src_exts = conver2id(unk, ext_word2id, sources)
    sources = conver2id(unk, word2id, sources)
    tar_ins = conver2id(unk, word2id, targets)
    targets = conver2id(unk, ext_word2id, targets)
    memories = conver2id(unk, word2id, memories)
    batch = list(zip(sources, src_exts, memories, tar_ins, targets))
    return batch

@curry
def convert_batch_copy_control(unk, word2id, batch):
    sources, targets, levels = map(list, unzip(batch))
    ext_word2id = dict(word2id)
    for source in sources:
        for word in source:
            if word not in ext_word2id:
                ext_word2id[word] = len(ext_word2id)
    src_exts = conver2id(unk, ext_word2id, sources)
    sources = conver2id(unk, word2id, sources)
    tar_ins = conver2id(unk, word2id, targets)
    targets = conver2id(unk, ext_word2id, targets)
    batch = list(zip(sources, src_exts, levels, tar_ins, targets))
    return batch

@curry
def convert_batch_extract_ptr(unk, word2id, emb_type, batch):
    def convert_one(sample):
        source_sents, extracts = sample
        if emb_type == "w2v":
            id_sents = conver2id(unk, word2id, source_sents)
        else:
            id_sents = [bert_tokenizer.convert_tokens_to_ids(sentence) for sentence in source_sents]
            #print("id_sents")
            #print(id_sents)
        #if len(id_sents) % 2 != 0:
        #    print("convert batch not 2!")
        #    print(len(source_sents))
        #    print(len(id_sents))
        return id_sents, extracts
    batch = list(map(convert_one, batch))
    return batch

@curry
def convert_batch_extract_ptr_backup(unk, word2id, emb_type, batch):
    def convert_one(sample):
        source_sents, extracts = sample
        if emb_type == "w2v":
            id_sents = conver2id(unk, word2id, source_sents)
        else:
            id_sents = [bert_tokenizer.convert_tokens_to_ids(sentence) for sentence in source_sents]
            print("source_sents")
            print(source_sents)
            print("id_sents")
            print(id_sents)
        #if len(id_sents) % 2 != 0:
        #    print("convert batch not 2!")
        #    print(len(source_sents))
        #    print(len(id_sents))
        return id_sents, extracts
    batch = list(map(convert_one, batch))
    return batch

@curry
def convert_batch_extract_ff(unk, word2id, batch):
    def convert_one(sample):
        source_sents, extracts = sample
        id_sents = conver2id(unk, word2id, source_sents)
        binary_extracts = [0] * len(source_sents)
        for ext in extracts:
            binary_extracts[ext] = 1
        return id_sents, binary_extracts
    batch = list(map(convert_one, batch))
    return batch


@curry
def pad_batch_tensorize(inputs, pad, cuda=True):
    """pad_batch_tensorize

    :param inputs: List of size B containing torch tensors of shape [T, ...]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    batch_size = len(inputs)
    max_len = max(len(ids) for ids in inputs)
    """
    except:
        print("inputs")
        print(inputs)
        print("pad")
        print(pad)
        print("cuda")
        print(cuda)
        exit()
    """
    tensor_shape = (batch_size, max_len)
    try:
        tensor = tensor_type(*tensor_shape)
    except:
        if all(len(inpt) == 0 for inpt in inputs):
            return None
        else:
            print("batch_size: {}".format(batch_size))
            print("max_len:{}".format(max_len))
            print("inputs: ")
            print(inputs)
            exit()
    tensor.fill_(pad)
    for i, ids in enumerate(inputs):
        tensor[i, :len(ids)] = tensor_type(ids)
    return tensor

@curry
def batchify_fn(pad, start, end, data, cuda=True):
    sources, targets = tuple(map(list, unzip(data)))

    src_lens = [len(src) for src in sources]
    tar_ins = [[start] + tgt for tgt in targets]
    targets = [tgt + [end] for tgt in targets]

    source = pad_batch_tensorize(sources, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)

    fw_args = (source, src_lens, tar_in)
    loss_args = (target, )
    return fw_args, loss_args


@curry
def batchify_fn_copy(pad, start, end, data, cuda=True):
    sources, ext_srcs, tar_ins, targets = tuple(map(list, unzip(data)))

    src_lens = [len(src) for src in sources]
    sources = [src for src in sources]
    ext_srcs = [ext for ext in ext_srcs]

    tar_ins = [[start] + tgt for tgt in tar_ins]
    targets = [tgt + [end] for tgt in targets]

    source = pad_batch_tensorize(sources, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)
    ext_src = pad_batch_tensorize(ext_srcs, pad, cuda)

    ext_vsize = ext_src.max().item() + 1
    fw_args = (source, src_lens, tar_in, ext_src, ext_vsize)
    loss_args = (target, )
    return fw_args, loss_args


@curry
def batchify_fn_copy_cond(pad, start, end, data, cuda=True):
    sources, ext_srcs, memories, tar_ins, targets = tuple(map(list, unzip(data)))

    src_lens = [len(src) for src in sources]
    sources = [src for src in sources]
    ext_srcs = [ext for ext in ext_srcs]

    tar_ins = [[start] + tgt for tgt in tar_ins]
    targets = [tgt + [end] for tgt in targets]

    memories = [mem for mem in memories]

    source = pad_batch_tensorize(sources, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)
    ext_src = pad_batch_tensorize(ext_srcs, pad, cuda)

    memory_lens = [len(mem) for mem in memories]
    memory = pad_batch_tensorize(memories, pad, cuda)

    ext_vsize = ext_src.max().item() + 1
    fw_args = (source, src_lens, memory, memory_lens, tar_in, ext_src, ext_vsize)
    loss_args = (target, )
    return fw_args, loss_args


@curry
def batchify_fn_copy_control(pad, start, end, data, cuda=True):
    sources, ext_srcs, levels, tar_ins, targets = tuple(map(list, unzip(data)))

    src_lens = [len(src) for src in sources]
    sources = [src for src in sources]
    ext_srcs = [ext for ext in ext_srcs]

    tar_ins = [[start] + tgt for tgt in tar_ins]
    targets = [tgt + [end] for tgt in targets]

    levels = [lev for lev in levels]

    source = pad_batch_tensorize(sources, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)
    ext_src = pad_batch_tensorize(ext_srcs, pad, cuda)

    level = torch.LongTensor(levels).to(source)

    ext_vsize = ext_src.max().item() + 1
    fw_args = (source, src_lens, level, tar_in, ext_src, ext_vsize)
    loss_args = (target, )
    return fw_args, loss_args


@curry
def batchify_fn_extract_ptr(pad, data, cuda=True):
    source_lists, targets = tuple(map(list, unzip(data)))

    src_nums = list(map(len, source_lists))
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))

    # PAD is -1 (dummy extraction index) for using sequence loss
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    remove_last = lambda tgt: tgt[:-1]
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda # use 0 here for feeding first conv sentence repr.
    )

    fw_args = (sources, src_nums, tar_in)
    loss_args = (target, )
    return fw_args, loss_args

@curry
def batchify_fn_extract_ff(pad, data, cuda=True):
    source_lists, targets = tuple(map(list, unzip(data)))

    src_nums = list(map(len, source_lists))
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))

    tensor_type = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    target = tensor_type(list(concat(targets)))

    fw_args = (sources, src_nums)
    loss_args = (target, )
    return fw_args, loss_args


def _batch2q(loader, prepro, q, single_run=True):
    epoch = 0
    while True:
        for batch in loader:
            q.put(prepro(batch))
        if single_run:
            break
        epoch += 1
        q.put(epoch)
    q.put(None)

class BucketedGenerater(object):
    def __init__(self, loader, prepro,
                 sort_key, batchify,
                 single_run=True, queue_size=8, fork=True):
        self._loader = loader
        self._prepro = prepro
        self._sort_key = sort_key
        self._batchify = batchify
        self._single_run = single_run
        if fork:
            ctx = mp.get_context('forkserver')
            self._queue = ctx.Queue(queue_size)
        else:
            # for easier debugging
            self._queue = None
        self._process = None

    def __call__(self, batch_size: int):
        def get_batches(hyper_batch):
            indexes = list(range(0, len(hyper_batch), batch_size))
            if not self._single_run:
                # random shuffle for training batches
                random.shuffle(hyper_batch)
                random.shuffle(indexes)
            hyper_batch.sort(key=self._sort_key)
            for i in indexes:
                batch = self._batchify(hyper_batch[i:i+batch_size])
                """
                except:
                    print("batch_size")
                    print(batch_size)
                    print("hyper_batch[i:i+batch_size]")
                    print(hyper_batch[i:i+batch_size])
                    print("convert!")
                    batch = convert_batch_extract_ptr_backup(1, {}, "bert", hyper_batch[i:i+batch_size])
                    print("batch after convert")
                    print(batch)
                    exit()
                """
                yield batch

        if self._queue is not None:
            ctx = mp.get_context('forkserver')
            self._process = ctx.Process(
                target=_batch2q,
                args=(self._loader, self._prepro,
                      self._queue, self._single_run)
            )
            self._process.start()
            while True:
                d = self._queue.get()
                if d is None:
                    break
                if isinstance(d, int):
                    print('\nepoch {} done'.format(d))
                    continue
                yield from get_batches(d)
            self._process.join()
        else:
            i = 0
            while True:
                for batch in self._loader:
                    yield from get_batches(self._prepro(batch))
                if self._single_run:
                    break
                i += 1
                print('\nepoch {} done'.format(i))

    def terminate(self):
        if self._process is not None:
            self._process.terminate()
            self._process.join()
