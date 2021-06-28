import os
from os.path import exists, join
import json
from utils import count_data
import argparse
import numpy as np


try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


def main(args):
    data_path = join(DATA_DIR, args.split)

    n_data = count_data(data_path)
    doc_sents_numbers = []
    sum_sents_numbers = []
    num_long_doc = 0

    for i in range(n_data):
        print('processing {}/{} ({:.2f}%%)\r'.format(i, n_data, 100 * i / n_data),
              end='')
        js_obj = json.load(open(join(data_path, "{}.json".format(i))))
        abstract = js_obj['abstract']
        article = js_obj['article']
        doc_sents_numbers.append(len(article))
        sum_sents_numbers.append(len(abstract))
        if len(article) > 400:
            num_long_doc += 1

    doc_sents_numbers = np.array(doc_sents_numbers)
    sum_sents_numbers = np.array(sum_sents_numbers)
    print()
    print("doc max: {}".format(doc_sents_numbers.max()))
    print("doc mean: {}".format(doc_sents_numbers.mean()))
    print("doc std: {}".format(doc_sents_numbers.std()))
    print()
    print("sum max: {}".format(sum_sents_numbers.max()))
    print("sum mean: {}".format(sum_sents_numbers.mean()))
    print("sum std: {}".format(sum_sents_numbers.std()))
    print()
    print("percent of long doc: {}".format(num_long_doc/n_data * 100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Output statistics')

    # choose metric to evaluate
    parser.add_argument('--split', action='store', required=True,
                        help='directory of decoded summaries')
    args = parser.parse_args()
    main(args)

