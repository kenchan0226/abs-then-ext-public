import os
from os.path import join, exists
import argparse
import json

"""
{ 
  'article_id': str,
  'abstract_text': List[str],
  'article_text': List[str],
  'section_names': List[str],
  'sections': List[List[str]]
}
"""


def main(file_name, out_data_dir, split):
    with open(file_name) as f_in:
        all_lines = f_in.readlines()

    os.makedirs(join(out_data_dir, split))
    num_processed_samples = 0

    for i, line in enumerate(all_lines):
        sample_obj = json.loads(line.strip())
        sample_obj["article"] = sample_obj["article_text"]
        del sample_obj["article_text"]
        #sample_obj["abstract"] = sample_obj["abstract_text"]
        #del sample_obj["abstract_text"]

        new_abstract = []
        for abs_sent in sample_obj["abstract_text"]:
            abs_sent = abs_sent.replace("<S>", "").replace("</S>", "").strip()
            new_abstract.append(abs_sent)
        sample_obj["abstract"] = new_abstract
        del sample_obj["abstract_text"]

        with open(join(out_data_dir, split, '{}.json'.format(i)), 'w') as f:
            json.dump(sample_obj, f, indent=4)
        num_processed_samples += 1

    print("Processed {} samples".format(num_processed_samples))
    print("Finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Make evaluation reference.'))
    parser.add_argument('-file_name', type=str, action='store',
                        help='The path of the data directory.')
    parser.add_argument('-out_data_dir', type=str, action='store',
                        help='The path of the data directory.')
    parser.add_argument('-split', type=str, action='store',
                        help='The folder name that needs to produce reference.')
    args = parser.parse_args()

    main(args.file_name, args.out_data_dir, args.split)
