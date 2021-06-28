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

split_list = ['test_cand_control_abs_2', 'test_cand_beam_len_2', 'test_cand_top2_beam', 'test_cand_2to1_0.15']

print("Rouge-L")

for split in split_list:
    print("split: {}".format(split))
    data_path = join(DATA_DIR, split)
    rouge_l_all = np.load(join(data_path, "rouge_l_all.dat"))
    #hist, bins = np.histogram(rouge_l_all, bins=[0, 0.25, 0.5, 0.75, 1], density=False)
    hist, bins = np.histogram(rouge_l_all, bins=[0, 0.2, 0.4, 0.6, 0.8, 1], density=False)
    print(hist)
    #print(" # of candidates > 0.5: {}".format())

print()
print("Rouge-1")

for split in split_list:
    print("split: {}".format(split))
    data_path = join(DATA_DIR, split)
    rouge_1_all = np.load(join(data_path, "rouge_1_all.dat"))
    #hist, bins = np.histogram(rouge_1_all, bins=[0, 0.25, 0.5, 0.75, 1], density=False)
    hist, bins = np.histogram(rouge_1_all, bins=[0, 0.2, 0.4, 0.6, 0.8, 1], density=False)
    print(hist)

print()
print("Rouge-2")

for split in split_list:
    print("split: {}".format(split))
    data_path = join(DATA_DIR, split)
    rouge_2_all = np.load(join(data_path, "rouge_2_all.dat"))
    #hist, bins = np.histogram(rouge_2_all, bins=[0, 0.25, 0.5, 0.75, 1], density=False)
    hist, bins = np.histogram(rouge_2_all, bins=[0, 0.2, 0.4, 0.6, 0.8, 1], density=False)
    print(hist)

