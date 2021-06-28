import argparse
import pickle as pkl
import os
from os.path import join
from tqdm import tqdm
import json
import re
from decoding import make_html_safe


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def main(pred_path, data_dir, split):
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
        os.makedirs(join(pred_path, 'output'))
    n_data = _count_data(join(data_dir, split))

    for i in range(n_data):
        js = json.load(open(join(data_dir, split, '{}.json'.format(i))))
        doc_sents = js["article"]
        lead_sents = []
        #word_limit = 200
        #word_count = 0
        word_left = 200
        for sent_i in doc_sents:
            if sent_i.strip() == "":
                continue
            sent_i_words = sent_i.split(" ")
            if word_left - len(sent_i_words) >= 0:
                lead_sents.append(sent_i)
                word_left -= len(sent_i_words)
            else:
                sent_i_trunc = " ".join(sent_i_words[:word_left])
                lead_sents.append(sent_i_trunc)
                break

        log = {'split': 'test'}
        json.dump(log, open(join(pred_path, 'log.json'), 'w'))
        with open(join(pred_path, 'output', '{}.dec'.format(i)), 'w') as f:
            f.write(make_html_safe('\n'.join(lead_sents)))


if __name__ == "__main__":
    data_dir = '../../datasets/pubmed_json'
    pred_path = 'decode_out/lead_200'
    split = 'test'
    main(pred_path, data_dir, split)
