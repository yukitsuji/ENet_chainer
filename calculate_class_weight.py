#!/usr/bin/env python3
# -*- coding: utf-8 -*
import argparse
import numpy as np
import sys
import cv2
import os
import matplotlib.pyplot
from enet.data_util.cityscapes.cityscapes_utils import cityscapes_labels

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="mean or log")
    parser.add_argument("--base_dir", type=str, default="./")
    parser.add_argument("--result", type=str, default="class_weight")
    parser.add_argument('--source', type=str, required=True,
                        help='absolute path to your data file')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='absolute path to your data file')
    parser.add_argument('--dataset', type=str, help="cityscapes or camvid")
    return parser.parse_args()

def calc_median_frequency(classes, present_num):
    """Class balancing by median frequency balancing method.
       Reference: https://arxiv.org/pdf/1411.4734.pdf
       'a = median_freq / freq(c) where freq(c) is the number of pixels
        of class c divided by the total number of pixels in images where
        c is present, and median_freq is the median of these frequencies.'
    """
    class_freq = classes / present_num
    median_freq = np.median(class_freq)
    return median_freq / class_freq

def calc_log_frequency(classes, value=1.02):
    """Class balancing by ENet method.
       prob = each_sum_pixel / each_sum_pixel.max()
       a = 1 / (log(1.02 + prob)).
    """
    class_freq = classes / classes.max() # LinkNet is max, but ENet is sum
    print(class_freq)
    print(np.log(value + class_freq))
    return 1 / np.log(value + class_freq)

def parse_cityscapes(label_orig):
    label_out = np.ones(label_orig.shape, dtype=np.int32) * -1
    for label in cityscapes_labels:
        if not label.ignoreInEval:
            label_out[label_orig == label.id] = label.trainId
    return label_out

if __name__ == '__main__':
    args = parse_arg()

    classes, present_num =\
        ([0 for i in range(args.num_classes)] for i in range(2))

    with open(args.source) as inf:
        for i, line in enumerate(inf):
            print('progress: {}'.format(i+1))
            columns = line.split()
            path = os.path.join(args.base_dir, columns[1][1:])
            path = path.replace("labelTrainIds", "labelIds")
            labels = cv2.imread(path, 0)
            if args.dataset == "cityscapes":
                labels = parse_cityscapes(labels)
            elif args.dataset == "camvid":
                raise NotImplementedError
            else:
                raise Exception("Please assign dataset to 'cityscapes or camvid'")

            for nc in range(args.num_classes):
                num_pixel = (labels == nc).sum()
                if num_pixel:
                    classes[nc] += num_pixel
                    present_num[nc] += 1

    if 0 in classes:
        raise Exception("Some classes are not found")

    classes = np.array(classes, dtype="f")
    presetn_num = np.array(classes, dtype="f")
    if args.method == "mean":
        class_weight = calc_median_frequency(classes, present_num)
    elif args.method == "log":
        class_weight = calc_log_frequency(classes)
    else:
        raise Exception("Please assign method to 'mean' or 'log'")

    print("class weight", class_weight)
    np.save(args.result, class_weight)
    print("Done!")
