""" make reference text files needed for ROUGE evaluation """
import json
import os
from os.path import join, exists
from time import time
from datetime import timedelta

from utils import count_data
from decoding import make_html_safe
import argparse

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


def dump(split):
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    dump_dir = join(DATA_DIR, 'refs', split)
    n_data = count_data(data_dir)
    for i in range(n_data):
        print('processing {}/{} ({:.2f}%%)\r'.format(i, n_data, 100*i/n_data),
              end='')
        with open(join(data_dir, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        abs_sents_list = data['abstract']
        for j, abs_sents in enumerate(abs_sents_list):
            with open(join(dump_dir, '{}.{}.ref'.format(chr(65+j), i)), 'w') as f:
                f.write(make_html_safe('\n'.join(abs_sents)))
    print('finished in {}'.format(timedelta(seconds=time()-start)))

def main(split):
    if split == 'all':
        for split in ['val', 'test']:  # evaluation of train data takes too long
            if not exists(join(DATA_DIR, 'refs', split)):
                os.makedirs(join(DATA_DIR, 'refs', split))
            dump(split)
    else:
        if not exists(join(DATA_DIR, 'refs', split)):
            os.makedirs(join(DATA_DIR, 'refs', split))
        dump(split)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Make evaluation reference.')
    )
    parser.add_argument('--folder_name', type=str, action='store', default='all',
                        help='The folder name that needs to produce reference. all means process both val and test.')
    args = parser.parse_args()

    main(args.folder_name)
