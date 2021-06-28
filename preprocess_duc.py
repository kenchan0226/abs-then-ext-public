from os.path import join
import json
import os
import random
import argparse
from collections import Counter, defaultdict
import pickle as pkl
import re
from bs4 import BeautifulSoup
import nltk


def sorted_nicely(l):
    """ From https://arcpy.wordpress.com/2012/05/11/sorting-alphanumeric-strings-in-python/
    Sorts the given iterable in the way that is expected.
    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def main(data_dir, out_dir):
    os.makedirs(out_dir)
    out_test_dir = join(out_dir, 'test')
    os.makedirs(out_test_dir)
    summary_dir = join(data_dir, "summaries")
    doc_dir = join(data_dir, "docs.with.sentence.breaks")
    doc_set_id_list = os.listdir(doc_dir)
    doc_set_id_list = sorted_nicely(doc_set_id_list)
    ref_doc_set_id_list = os.listdir(summary_dir)
    num_doc = 0
    num_ref = 0
    for doc_set_id in doc_set_id_list:
        regex = re.compile('{}.'.format(doc_set_id))
        ref_dirs = [ref_dir for ref_dir in ref_doc_set_id_list if re.match(regex, ref_dir)]
        doc_id_list = os.listdir(join(doc_dir, doc_set_id))
        doc_id_list = sorted_nicely(doc_id_list)
        doc_set_summary_dict = defaultdict(list)

        for ref_dir in ref_dirs:
            if os.path.exists(join(summary_dir, ref_dir, 'perdocs')):
                with open(join(summary_dir, ref_dir, 'perdocs')) as f:
                    ref_html_txt = f.read()
                ref_soup = BeautifulSoup(ref_html_txt, 'lxml')
                all_doc_summary_list = ref_soup.find_all('sum')

                for doc_summary in all_doc_summary_list:
                    num_ref += 1
                    doc_set_summary_dict[doc_summary.attrs['docref']].append(nltk.tokenize.sent_tokenize(doc_summary.text.strip().replace('\n', ' ')))

        for doc_id in doc_id_list:
            # doc_id with ".s"
            with open(join(doc_dir, doc_set_id, doc_id)) as f:
                html_txt = f.read()
            doc_soup = BeautifulSoup(html_txt, 'lxml')
            raw_s_list = doc_soup.find('text').find_all('s')
            doc_sentence_list = [s.text.strip() for s in raw_s_list]
            doc_summary_list = doc_set_summary_dict[doc_id[:-2]]

            json_out = {"article": doc_sentence_list, "abstract": doc_summary_list, "doc_ref": doc_id[:-2], "doc_set": doc_set_id}
            with open(join(out_test_dir, '{}.json'.format(num_doc)), 'w') as f:
                json.dump(json_out, f, indent=4)
            num_doc += 1

    print("Total no. of doc: {}".format(num_doc))
    print("Total no. of references: {}".format(num_ref))


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
