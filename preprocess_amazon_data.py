import json
import os
from os.path import join
import re
from collections import Counter
import argparse


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def extract_source(data_dir, split, out_dir):
    n_data = _count_data(join(data_dir, split))
    os.makedirs(join(out_dir, split))

    for i in range(n_data):
        js = json.load(open(join(data_dir, split, '{}.json'.format(i))))
        js['article'] = js['reviewText']
        js['abstract'] = js['summary']
        js.pop('reviewText', None)
        js.pop('summary', None)

        with open(join(out_dir, split, '{}.json'.format(i)), 'w') as f:
            json.dump(js, f, indent=4)

    print("Finished split: {}".format(split))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Make extraction labels')
    )
    parser.add_argument('-data', type=str, action='store',
                        help='The directory of the data.')
    parser.add_argument('-out_dir', type=str, action='store', default='all',
                        help='The folder name that needs to produce candidates. all means process both train and val.')
    args = parser.parse_args()

    os.makedirs(args.out_dir)

    splits = ['train', 'val', 'test']
    for split in splits:
        extract_source(args.data, split, args.out_dir)
