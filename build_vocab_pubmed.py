from os.path import join
import os
import argparse
from collections import Counter
import pickle as pkl


def main(data_dir):
    with open(join(data_dir, "vocab")) as f_in:
        all_lines = f_in.readlines()

    vocab_counter = Counter()

    for i, line in enumerate(all_lines):
        word, count = line.strip().split(" ")
        vocab_counter[word] = int(count)
        #print(word)
        #print(int(count))
        #print(vocab_counter)
        #exit()

    with open(os.path.join(data_dir, "vocab_cnt.pkl"),
              'wb') as vocab_file:
        pkl.dump(vocab_counter, vocab_file, protocol=4)
    print("Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Preprocess review data')
    )
    parser.add_argument('--data_dir', type=str, action='store',
                        help='The directory of the data.')
    args = parser.parse_args()
    main(args.data_dir)
