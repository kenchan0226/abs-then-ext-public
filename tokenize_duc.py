from os.path import join
import json
from nltk.parse import CoreNLPParser
import os
import random
import argparse
from collections import Counter, defaultdict
import pickle as pkl
import re
from bs4 import BeautifulSoup
import nltk

corenlp_parser = CoreNLPParser(url='http://localhost:9000')

def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def main(data_dir, out_dir):
    split_dir = join(data_dir, 'test')
    os.makedirs(out_dir)
    out_test_dir = join(out_dir, 'test')
    os.makedirs(out_test_dir)
    n_data = _count_data(split_dir)
    for i in range(n_data):
        js = json.load(open(join(split_dir, '{}.json'.format(i))))

        document = js['article']
        tokenized_document = [' '.join(corenlp_parser.tokenize(doc_sent.strip())) for doc_sent in document]
        summary_list = js['abstract']

        tokenized_summary_list = []
        for summary in summary_list:
            tokenized_summary = [' '.join(corenlp_parser.tokenize(summary_sent.strip())) for summary_sent in summary]
            tokenized_summary_list.append(tokenized_summary)
        js['article'] = tokenized_document
        js['abstract'] = tokenized_summary_list

        with open(join(out_test_dir, '{}.json'.format(i)), 'w') as f:
            json.dump(js, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Preprocess duc data')
    )
    parser.add_argument('--data_dir', type=str, action='store',
                        help='The directory of the data.')
    parser.add_argument('--out_dir', type=str, action='store',
                        help='The output directory of the data.')
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)

