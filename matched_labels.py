import os
from os.path import join
import argparse
import re
import json
import numpy as np


try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def main(args):
    data_path = join(DATA_DIR, args.split)
    n_data = _count_data(data_path)
    num_match = 0
    num_ref_sents = 0

    for i in range(n_data):
        js_obj = json.load(open(join(data_path, "{}.json".format(i))))
        with open(join(args.dec_folder, "output/{}.dec".format(i))) as f:
            dec_lines = f.readlines()
        num_ref_sents += len(js_obj['extracted'])
        for sent_i, ext_label in enumerate(js_obj['extracted']):
            ref_sent = js_obj['article'][ext_label].strip()
            if sent_i < len(dec_lines):
                dec_sent = dec_lines[sent_i].strip()
                if ref_sent == dec_sent:
                    num_match += 1

    print("percentage of matched: {:.3f}".format(num_match/num_ref_sents))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Output statistics')

    # choose metric to evaluate
    parser.add_argument('--split', action='store', required=True,
                        help='directory of ground-truth summaries')
    parser.add_argument('--dec_folder', action='store', required=True,
                        help='directory of decoded summaries')
    args = parser.parse_args()
    main(args)

