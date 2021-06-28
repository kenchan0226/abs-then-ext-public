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
    compress_ratios = []
    compress_sents_lens = []
    original_sents_lens = []
    ext_label_rouge_l_scores = []
    distances_two_to_one = []
    improvements_all = []

    for i in range(n_data):
        js_obj = json.load(open(join(data_path, "{}.json".format(i))))
        abstract = js_obj['abstract']
        article = js_obj['article']
        ext_ids = js_obj['extracted']
        rouge_l_scores = js_obj['score']
        #computed_compress_ratios = js_obj['compression_ratios']
        assert len(rouge_l_scores) == len(ext_ids)
        distances_two_to_one += [d for d in js_obj['distances_two_to_one_{}'.format(args.threshold)] if d is not None]  # filter out all none distance

        for j, ext_id in enumerate(ext_ids):
            original_sent = article[ext_id].strip().split(' ')
            compressed_sent = abstract[j].strip().split(' ')
            original_sent_len = len(original_sent)
            compressed_sent_len = len(compressed_sent)
            compress_ratio = (original_sent_len - compressed_sent_len) / original_sent_len
            #if abs(compress_ratio-computed_compress_ratios[j]) > 1e-4:
            #    raise ValueError
            compress_ratios.append(compress_ratio)
            compress_sents_lens.append(compressed_sent_len)
            original_sents_lens.append(original_sent_len)
            ext_label_rouge_l_scores.append(rouge_l_scores[j])

        improvements_all += js_obj['improvements']

    compress_ratios = np.array(compress_ratios)
    compress_sents_lens = np.array(compress_sents_lens)
    original_sents_lens = np.array(original_sents_lens)
    ext_label_rouge_l_scores = np.array(ext_label_rouge_l_scores)
    distances_two_to_one = np.array(distances_two_to_one)
    improvements_all = np.array(improvements_all)

    compress_ratios.dump(join(data_path, "compress_ratios.dat"))
    compress_sents_lens.dump(join(data_path, "compress_sents_lens.dat"))
    original_sents_lens.dump(join(data_path, "original_sents_lens.dat"))
    ext_label_rouge_l_scores.dump(join(data_path, "ext_label_rouge_l_scores.dat"))
    distances_two_to_one.dump(join(data_path, "distances_two_to_one_{}.dat".format(args.threshold)))
    improvements_all.dump(join(data_path, "improvements.dat"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Output statistics')

    # choose metric to evaluate
    parser.add_argument('--split', action='store', required=True,
                        help='directory of decoded summaries')
    parser.add_argument('--threshold', type=float, action='store', required=True,
                        help='threshold of the two to one extraction label')
    args = parser.parse_args()
    main(args)

