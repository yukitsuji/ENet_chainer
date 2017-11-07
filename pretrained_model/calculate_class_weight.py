#!/usr/bin/env python3
# -*- coding: utf-8 -*
import argparse
import numpy as np
import sys
import cv2
import os

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="mean or log")
    parser.add_argument("--base_dir", type=str, default="./")
    parser.add_argument("--result", type=str, default="class_weight")
    parser.add_argument('--source', type=str, required=True,
                        help='absolute path to your data file')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='absolute path to your data file')
    return parser.parse_args()

def calc_median_frequency(classes):
    class_freq = np.array(classes)
    median_freq = np.mean(class_freq)
    return median_freq / class_freq

def calc_log_frequency(classes, value=1.02):
    class_freq = np.array(classes)
    class_freq /= class_freq.sum()
    return 1 / np.log(value + class_freq)

if __name__ == '__main__':
    args = parse_arg()

    classes = [0 for i in range(args.num_classes)]

    with open(args.source) as inf:
        for i, line in enumerate(inf):
            print('progress: {}'.format(i+1))
            columns = line.split()
            path = os.path.join(args.base_dir, columns[1][1:])
            path = path.replace("labelTrainIds", "labelIds")
            labels = cv2.imread(path, 0)
            for nc in range(args.num_classes):
                classes[nc] += (labels == nc).sum()

    if 0 in classes:
        raise Exception("Some classes are not found")

    if args.method == "mean"
        class_weight = calc_median_frequency(classes)
    elif args.method == "log":
        class_weight = calc_log_frequency(classes)

    np.save(args.result, class_weight)
    print("Done!")
