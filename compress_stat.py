import os
from os.path import join
import argparse
import re
import json
import numpy as np
from matplotlib import pyplot as plt


try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


def main(args):
    data_path = join(DATA_DIR, args.split)
    compress_ratios = np.load(join(data_path, "compress_ratios.dat"))
    # less train/compress_ratios.dat

    compress_sents_lens = np.load(join(data_path, "compress_sents_lens.dat"))
    original_sents_lens = np.load(join(data_path, "original_sents_lens.dat"))
    ext_label_rouge_l_scores = np.load(join(data_path, "ext_label_rouge_l_scores.dat"))
    distances_one_to_many = np.load(join(data_path, "distances_two_to_one_{}.dat".format(args.threshold)))
    improvements = np.load(join(data_path, "improvements.dat"))

    #print(compress_ratios.shape)
    #print(original_sents_lens.shape)
    #print(compress_sents_lens.shape)
    #print(ext_label_rouge_l_scores.shape)
    num_ext_label = compress_ratios.shape[0]

    print("max compression lengths:\t{}".format(max(compress_sents_lens)))
    print("mean compression lengths:\t{:.3f}".format((compress_sents_lens.mean())))
    print("min compression lengths:\t{}".format(min(compress_sents_lens)))
    print("max original lengths:\t{}".format(max(original_sents_lens)))
    print("mean original lengths:\t{:.3f}".format((original_sents_lens.mean())))
    print("min original lengths:\t{}".format(min(original_sents_lens)))
    print("max compression ratio:\t{:.3f}".format(max(compress_ratios)))
    print("mean compression ratio:\t{:.3f}".format((compress_ratios.mean())))
    print("min compression ratio:\t{:.3f}".format(min(compress_ratios)))
    print("max Rouge-L score:\t{}".format(max(ext_label_rouge_l_scores)))
    print("mean Rouge-L score:\t{:.3f}".format((ext_label_rouge_l_scores.mean())))
    print("min Rouge-L score:\t{}".format(min(ext_label_rouge_l_scores)))

    print("histogram of compression ratio:")
    # [0, 0.2, 0.4, 0.6, 0.8, 1]
    compress_ratios_weights = np.ones_like(compress_ratios) / compress_ratios.shape[0]
    #hist, bins = np.histogram(compress_ratios, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], density=False)
    hist, bins = np.histogram(compress_ratios, bins=[0, 0.4, 0.65, 1], density=False)
    plt.figure()
    plt.hist(compress_ratios, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], density=False, weights=compress_ratios_weights)
    plt.title("Histogram of compression ratio in NYT")
    plt.xlabel('Compression ratio')
    plt.ylabel('Proportion of samples')
    #plt.show()
    plt.savefig(join(data_path, "compress_ratio_hist.pdf"))
    plt.savefig(join(data_path, "compress_ratio_hist.png"))
    print(hist)
    print(bins)
    #print(sum(hist))

    print("histogram of Rouge-L scores of extraction label:")
    ext_label_rouge_l_scores_weights = np.ones_like(ext_label_rouge_l_scores) / ext_label_rouge_l_scores.shape[0]
    hist, bins = np.histogram(ext_label_rouge_l_scores, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], density=False)
    plt.figure()
    plt.hist(ext_label_rouge_l_scores, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], density=False, weights=ext_label_rouge_l_scores_weights)
    plt.title("Histogram of Rouge-l scores of extraction label in CNN/DM")
    plt.xlabel('Rouge-l score')
    plt.ylabel('Proportion of samples')
    #plt.show()
    plt.savefig(join(data_path, "ext_label_rouge_l.pdf"))
    plt.savefig(join(data_path, "ext_label_rouge_l.png"))
    print(hist)
    print(bins)
    #print(sum(hist))

    print("Ext label with Rouge_l less than 0.1:\t{:.3f}".format(hist[0]/num_ext_label))
    print("Ext label with Rouge_l less than 0.2:\t{:.3f}".format( (hist[0] + hist[1]) / num_ext_label) )
    print("Ext label with Rouge_l less than 0.3:\t{:.3f}".format((hist[0] + hist[1] + hist[2]) / num_ext_label))

    print("number of two to one extraciton labels:\t{}".format(distances_one_to_many.shape[0]))
    print("number of summary sents:\t{}".format(num_ext_label))
    print("max two to one distance:\t{}".format(max(distances_one_to_many)))
    print("average two to one distance:\t{:.3f}".format(sum(distances_one_to_many) / distances_one_to_many.shape[0]))
    print("min two to one distance:\t{}".format(min(distances_one_to_many)))
    print("histogram of two to one distance:")
    hist, bins = np.histogram(distances_one_to_many, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
                              density=False)
    print(hist)
    print(bins)

    print("hist. of improvements")
    hist, bins = np.histogram(improvements, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], density=False)
    print(hist)
    print(bins)

    print(hist[0]/num_ext_label)
    print((hist[0] + hist[1]) / num_ext_label)
    print((hist[0] + hist[1] + hist[2]) / num_ext_label)

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

