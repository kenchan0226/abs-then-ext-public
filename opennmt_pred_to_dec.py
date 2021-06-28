from os.path import join
import json
import os
import random
import argparse
from collections import Counter
import pickle as pkl
import re
import nltk
from decoding import make_html_safe


def main(pred_path, out_dir):
    with open(pred_path) as f_in:
        pred_lines = f_in.readlines()
    os.makedirs(out_dir)
    out_dec_dir = join(out_dir, 'output')
    os.makedirs(out_dec_dir)
    num_exported_samples = 0
    for pred_summary in pred_lines:
        pred_summary = pred_summary.strip().replace('<t>', '').replace('</t>', '').strip()
        pred_summary_sent_list = nltk.tokenize.sent_tokenize(pred_summary)
        with open(join(out_dec_dir, '{}.dec'.format(num_exported_samples)), 'w') as f:
            f.write(make_html_safe('\n'.join(pred_summary_sent_list)))
        num_exported_samples += 1

    log_dict = {"split": "test"}

    with open(join(out_dir, 'log.json'), 'w') as f:
        json.dump(log_dict, f, indent=4)

    print("Decoded {} samples.".format(num_exported_samples))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Preprocess review data')
    )
    parser.add_argument('--pred', type=str, action='store',
                        help='The directory of the opennmt pred file.')
    parser.add_argument('--out_dir', type=str, action='store',
                        help='The directory of the output data.')
    args = parser.parse_args()
    main(args.pred, args.out_dir)
