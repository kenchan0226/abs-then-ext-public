""" RL training utilities"""
import math
from time import time
from datetime import timedelta

from toolz.sandbox.core import unzip
from cytoolz import concat

import numpy as np
import torch
from torch.nn import functional as F
from torch import autograd
from torch.nn.utils import clip_grad_norm_

from metric import compute_rouge_l, compute_rouge_n
from training import BasicPipeline
import sys

#from decoding import make_html_safe
from data.batcher import bert_tokenizer


def a2c_validate(agent, abstractor, loader, disable_selected_mask=False, is_conditional_abs=False):
    agent.eval()
    start = time()
    print('start running validation...', end='')
    avg_reward = 0
    i = 0

    # debug
    #extracted_local_idx_2dlist = []
    #num_batches = 0

    with torch.no_grad():
        for art_batch, raw_art_batch, raw_abs_batch in loader:

            # debug
            #num_batches += 1

            num_articles = len(art_batch)
            num_ext_sents = 0
            if is_conditional_abs:
                sequential_ext_sents = []
                sequential_article_ids = []
            else:
                ext_sents = []
            ext_inds = []
            for article_i, raw_arts_tokenized in enumerate(art_batch):

                # debug
                """
                if num_batches == 2 and article_i == 27:
                    print("disable_selected_mask")
                    print(disable_selected_mask)
                    print("raw_arts_tokenized")
                    print(raw_arts_tokenized)
                    exit()
                """

                indices = agent(raw_arts_tokenized, disable_selected_mask)
                ext_inds += [(num_ext_sents, len(indices)-1)]
                num_ext_sents += len(indices) - 1

                if is_conditional_abs:
                    # insert place holder to sequential_ext_sents
                    num_selected_sents_excluded_eos = len(indices) - 1
                    if num_selected_sents_excluded_eos > len(sequential_ext_sents):
                        [sequential_ext_sents.append([]) for _ in
                         range(num_selected_sents_excluded_eos - len(sequential_ext_sents))]
                        [sequential_article_ids.append([]) for _ in
                         range(num_selected_sents_excluded_eos - len(sequential_article_ids))]

                    for idx_i, idx in enumerate(indices):
                        if idx.item() < len(raw_arts_tokenized):
                            # ext_sents.append(raw_arts_tokenized[idx.item()])
                            sequential_ext_sents[idx_i].append(raw_arts_tokenized[idx.item()])
                            sequential_article_ids[idx_i].append(article_i)
                else:
                    ext_sents += [raw_art_batch[article_i][idx.item()] for idx in indices if idx.item() < len(raw_arts_tokenized)]

                    # debug
                    #extracted_local_idx_2dlist.append([idx.item() for idx in indices if idx.item() < len(raw_arts_tokenized)])

            # abstract
            if is_conditional_abs:
                all_summs = abstractor(sequential_ext_sents, sequential_article_ids, num_articles)
            else:
                all_summs = abstractor(ext_sents)

            for (j, n), abs_sents in zip(ext_inds, raw_abs_batch):
                summs = all_summs[j:j+n]
                # python ROUGE-1 (not official evaluation)
                avg_reward += compute_rouge_n(list(concat(summs)),
                                              list(concat(abs_sents)), n=1)

                i += 1
    avg_reward /= (i/100)
    print('finished in {}! avg reward: {:.2f}'.format(
        timedelta(seconds=int(time()-start)), avg_reward))

    # debug
    #extracted_local_idx_2darray = np.array(extracted_local_idx_2dlist)
    #extracted_local_idx_2darray.dump('/home/ubuntu/ken/projects/abstract_then_extract/val_selected_indices_2d.dat')

    return {'reward': avg_reward}


def a2c_train_step(agent, abstractor, loader, opt, grad_fn,
                   gamma=0.99, reward_fn=compute_rouge_l,
                   stop_reward_fn=compute_rouge_n(n=1), stop_coeff=1.0, reward_type=0, disable_selected_mask=False, is_conditional_abs=False, debug=False):
    #print('a2c train step')
    #sys.stdout.flush()
    opt.zero_grad()
    indices = []
    probs = []
    baselines = []
    if is_conditional_abs:
        sequential_ext_sents = []
        sequential_article_ids = []
    else:
        ext_sents = []
    art_batch, raw_art_batch, raw_abs_batch = next(loader)
    #print('Loader next')
    #sys.stdout.flush()
    num_articles = len(art_batch)
    # extract
    for article_i, raw_arts_tokenized in enumerate(art_batch):  # extract sent indices for each article
        """
        if debug:
            print("raw_arts_tokenized[0:5]")
            print(" ".join(raw_arts_tokenized[0]))
            print(" ".join(raw_arts_tokenized[1]))
            print(" ".join(raw_arts_tokenized[2]))
            print(" ".join(raw_arts_tokenized[3]))
        """
        (inds, ms), bs = agent(raw_arts_tokenized, disable_selected_mask, debug=debug)
        """
        if debug:
            print("inds length: {}".format(len(inds)))
            print("ms shape: {}".format(len(ms)))
            print("bs shape: {}".format(len(bs)))
        """

        baselines.append(bs)
        indices.append(inds)
        probs.append(ms)


        if is_conditional_abs:
            # insert place holder to sequential_ext_sents
            num_selected_sents_excluded_eos = len(inds) - 1
            if num_selected_sents_excluded_eos > len(sequential_ext_sents):
                [sequential_ext_sents.append([]) for _ in range( num_selected_sents_excluded_eos - len(sequential_ext_sents) )]
                [sequential_article_ids.append([]) for _ in range( num_selected_sents_excluded_eos - len(sequential_article_ids) )]

            for i, idx in enumerate(inds):
                if idx.item() < len(raw_arts_tokenized):
                    #ext_sents.append(raw_arts_tokenized[idx.item()])
                    sequential_ext_sents[i].append(raw_arts_tokenized[idx.item()])
                    sequential_article_ids[i].append(article_i)

        else:
            ext_sents += [raw_art_batch[article_i][idx.item()]
                          for idx in inds if idx.item() < len(raw_arts_tokenized)]

    # abstract
    with torch.no_grad():
        if is_conditional_abs:
            summaries = abstractor(sequential_ext_sents, sequential_article_ids, num_articles)
        else:
            summaries = abstractor(ext_sents)

    i = 0
    rewards = []
    avg_reward = 0
    #reward_lens = []
    #print("len indices and batch")
    #print(len(indices))
    #print(len(abs_batch))
    #print()
    #print()
    for inds, abss in zip(indices, raw_abs_batch):
        # process each article
        if reward_type == 0:
            rs = ([reward_fn(summaries[i+j], abss[j]) for j in range(min(len(inds)-1, len(abss)))]
                  + [0 for _ in range(max(0, len(inds)-1-len(abss)))]
                  + [stop_coeff*stop_reward_fn(
                      list(concat(summaries[i:i+len(inds)-1])),
                      list(concat(abss)))])
        elif reward_type == 1:
            sent_reward = [reward_fn(summaries[i:i+j], abss) for j in range(len(inds)-1)]
            shaped_reward = [sent_reward[0]] + [sent_reward[i+1] - sent_reward[i] for i in range(0, len(sent_reward)-1)] if len(sent_reward) > 0 else []

            rs = shaped_reward + [stop_coeff*stop_reward_fn(
                      list(concat(summaries[i:i+len(inds)-1])),
                      list(concat(abss)))]
            #print("rs: {}".format(rs))
        elif reward_type == 2 or reward_type == 3 or reward_type == 4:
            # debug
            #print("abss")
            #print(abss)
            #print()
            #print("prediction")
            #print([ (i,i + j) for j in range(min(len(inds) - 1, len(abss)))])
            #print( [summaries[i:i + j] for j in range( min(len(inds)-1, len(abss)) )] )
            #print()

            sent_reward = [reward_fn(summaries[i:i + j + 1], abss) for j in range( min(len(inds)-1, len(abss)) )]
            shaped_reward = [sent_reward[0]] + [sent_reward[i + 1] - sent_reward[i] for i in range(0, len(sent_reward) - 1)] if len(sent_reward) > 0 else []

            # debug
            #print("sent_reward")
            #print(sent_reward)
            #print("shaped_reward")
            #print(shaped_reward)
            #print()

            rs = shaped_reward + [0 for _ in range(max(0, len(inds)-1-len(abss)))] + [stop_coeff*stop_reward_fn(
                      list(concat(summaries[i:i+len(inds)-1])),
                      list(concat(abss)))]
        else:
            raise ValueError
        #print("inds: {}".format(inds))
        #print("rs: {}".format(rs))
        #if len(inds) - 1 == 0:
        #    print("rs: {}".format(rs))
        #    print("stop reward: {}".format([stop_coeff*stop_reward_fn(
        #              list(concat(summaries[i:i+len(inds)-1])),
        #              list(concat(abss)))]))

        assert len(rs) == len(inds)
        #raise ValueError

        avg_reward += rs[-1]/stop_coeff
        i += len(inds)-1
        # compute discounted rewards
        R = 0
        disc_rs = []  # a list of discounted reward, with len=len(inds)
        for r in rs[::-1]:
            R = r + gamma * R
            disc_rs.insert(0, R)
        #reward_lens.append(len(disc_rs))
        rewards += disc_rs  # a list of all discounted reward in the batch.

    # baselines
    """
    print("length of reward lens")
    print(len(reward_lens))
    print(reward_lens)
    print()
    print("baselines")
    print(len(baselines))
    print()
    for b in baselines:
        print(len(b))
    print()
    print("inds")
    for inds in indices:
        print(len(inds))
    """

    indices = list(concat(indices))
    probs = list(concat(probs))
    baselines = list(concat(baselines))
    # standardize rewards
    reward = torch.Tensor(rewards).to(baselines[0].device)
    reward = (reward - reward.mean()) / (
        reward.std() + float(np.finfo(np.float32).eps))
    baseline = torch.cat(baselines).squeeze()
    avg_advantage = 0
    losses = []
    for action, p, r, b in zip(indices, probs, reward, baseline):
        advantage = r - b
        avg_advantage += advantage
        losses.append(-p.log_prob(action)
                      * (advantage/len(indices))) # divide by T*B
    critic_loss = F.mse_loss(baseline, reward)
    # backprop and update
    autograd.backward(
        [critic_loss] + losses,
        [torch.tensor(1.0).to(critic_loss.device)] + [torch.ones(1).to(critic_loss.device)] * (len(losses))
    )
    grad_log = grad_fn()
    opt.step()
    log_dict = {}
    log_dict.update(grad_log)
    log_dict['reward'] = avg_reward/len(art_batch)
    log_dict['advantage'] = avg_advantage.item()/len(indices)
    log_dict['mse'] = critic_loss.item()
    assert not math.isnan(log_dict['grad_norm'])
    return log_dict


def get_grad_fn(agent, clip_grad, max_grad=1e2):
    """ monitor gradient for each sub-component"""
    params = [p for p in agent.parameters()]
    def f():
        grad_log = {}
        for n, m in agent.named_children():
            tot_grad = 0
            for p in m.parameters():
                if p.grad is not None:
                    tot_grad += p.grad.norm(2) ** 2
            tot_grad = tot_grad ** (1/2)
            #grad_log['grad_norm'+n] = tot_grad.item()
            grad_log['grad_norm' + n] = tot_grad
        grad_norm = clip_grad_norm_(
            [p for p in params if p.requires_grad], clip_grad)
        grad_norm = grad_norm
        #grad_norm = grad_norm.item()
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log['grad_norm'] = grad_norm
        return grad_log
    return f


class A2CPipeline(BasicPipeline):
    def __init__(self, name,
                 net, abstractor,
                 train_batcher, val_batcher,
                 optim, grad_fn,
                 reward_fn, gamma,
                 stop_reward_fn, stop_coeff, reward_type, disable_selected_mask=False, is_conditional_abs=False, debug=False):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._opt = optim
        self._grad_fn = grad_fn
        self._abstractor = abstractor
        self._gamma = gamma
        self._reward_fn = reward_fn
        self._stop_reward_fn = stop_reward_fn
        self._stop_coeff = stop_coeff
        self._disable_selected_mask = disable_selected_mask
        self._n_epoch = 0  # epoch not very useful?
        self._reward_type = reward_type
        self.debug = debug
        self._is_conditional_abs = is_conditional_abs

    def batches(self):
        raise NotImplementedError('A2C does not use batcher')

    def train_step(self):
        # forward pass of model
        self._net.train()
        #print('train step')
        #sys.stdout.flush()
        log_dict = a2c_train_step(
            self._net, self._abstractor,
            self._train_batcher,
            self._opt, self._grad_fn,
            self._gamma, self._reward_fn,
            self._stop_reward_fn, self._stop_coeff, self._reward_type, self._disable_selected_mask, self._is_conditional_abs, self.debug
        )
        return log_dict

    def validate(self):
        return a2c_validate(self._net, self._abstractor, self._val_batcher, self._disable_selected_mask, self._is_conditional_abs)

    def checkpoint(self, *args, **kwargs):
        # explicitly use inherited function in case I forgot :)
        return super().checkpoint(*args, **kwargs)

    def terminate(self):
        pass  # No extra processs so do nothing
