import os
from os.path import exists, join
import json
from utils import count_data
import argparse


try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


def main(args):
    data_path = join(DATA_DIR, args.split)

    n_data = count_data(data_path)
    compress_ratios = []
    compress_sents_lens = []
    original_sents_lens = []

    for i in range(n_data):
        print('processing {}/{} ({:.2f}%%)\r'.format(i, n_data, 100 * i / n_data),
              end='')
        js_obj = json.load(open(join(data_path, "{}.json".format(i))))
        abstract = js_obj['abstract']
        article = js_obj['article']
        ext_ids = js_obj['extracted']

        for j, ext_id in enumerate(ext_ids):
            original_sent = article[ext_id].strip().split(' ')
            compressed_sent = abstract[j].strip().split(' ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Output statistics')

    # choose metric to evaluate
    parser.add_argument('--split', action='store', required=True,
                        help='directory of decoded summaries')
    args = parser.parse_args()
    main(args)

