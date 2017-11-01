#!/usr/env/bin python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import json
import numpy as np
import os
import subprocess
import shutil
import yaml

try:
    import cupy as cp
except:
    cp = None
    print("Please install cupy if you want to use gpus")

from sklearn.model_selection import train_test_split

import chainer
from chainer import iterators
from chainer.training import extensions

from models import enet_paper

from collections import OrderedDict
yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))

# from iou_tracker.models import rnn_predictor
# from iou_tracker.models import da_predictor
# from iou_tracker.models import siamese_predictor
# from iou_tracker.data_util import mot_loader, cuhk_loader
# from iou_tracker.network_util import updater, siamese_updater
# from iou_tracker.network_util import evaluator, siamese_evaluator
# from iou_tracker.data_util.utils import split_dataset, split_dataset_4


SEED = 0

def parse_dict(dic, key, value=None):
    return value if not key in dic else dic[key]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='default.yml', type=str, help='configure file')
    args = parser.parse_args()
    config = yaml.load(open(args.config))

    SEED = parse_dict(config, "seed", 0)
    np.random.seed(SEED)
    if cp:
        cp.random.seed(SEED)

    if config["mode"] == "Test":
        chainer.global_config.train = False
        chainer.global_config.enable_backprop = False

    subprocess.check_call(["mkdir", "-p", config["results"]])
    shutil.copy(args.config, os.path.join(config['results'], args.config.split('/')[-1]))
    return config

def parse_trigger(trigger):
    return (int(trigger[0]), trigger[1])

def create_extension(trainer, test_iter, model, config):
    """Create extension for training models"""
    for key, ext in config.items():
        if key == "Evaluator":
            parent_cls = siamese_evaluator if ext['type'] == 'siamese' else evaluator
            cl = getattr(parent_cls, ext['name'])
            args = parse_dict(ext, 'args', {})
            trainer.extend(cl(
                test_iter, model, **args))
        elif key == "dump_graph":
            cl = getattr(extensions, key)
            trainer.extend(cl(ext['name']))
        elif key == 'snapshot':
            cl = getattr(extensions, key)
            trigger = parse_trigger(ext['trigger'])
            trainer.extend(cl(), trigger=trigger)
        elif key == 'LogReport':
            cl = getattr(extensions, key)
            trigger = parse_trigger(ext['trigger'])
            trainer.extend(cl(trigger=trigger))
        elif key == "PrintReport":
            cl = getattr(extensions, key)
            report_list = ext['name'].split(' ')
            trigger = parse_trigger(ext['trigger'])
            trainer.extend(cl(report_list), trigger=trigger)
        elif key == "ProgressBar":
            cl = getattr(extensions, key)
            trainer.extend(cl(update_interval=ext['update_interval']))
    return trainer

def create_updater(train_iter, optimizer, config, args):
    parent_cls = siamese_updater if config['type'] == 'siamese' else updater
    Updater = getattr(parent_cls, config['name'])
    dic = parse_dict(config, 'args', {})
    dic['models'] = optimizer.target
    dic.update(args)
    return Updater(train_iter, optimizer, **dic)

def create_optimizer(config, model):
    Optimizer = getattr(chainer.optimizers, config['name'])
    opt = Optimizer(**config['args'])
    opt.setup(model)
    return opt

def create_iterator(train_data, test_data, config):
    Iterator = getattr(chainer.iterators, config['name'])
    train_iter = Iterator(train_data, config['batchsize'])
    test_iter = Iterator(test_data, config['batchsize'], repeat=False)
    return train_iter, test_iter

def parse_size(mode):
    if mode == "vgg":
        return (224, 224)
    if mode == 'small':
        return (16, 32)
    if mode == 'bigger':
        return (32, 64)
    return None

def load_rnn_dataset(config):
    """Load specified dataset and split it to train/val."""
    loader = getattr(mot_loader, config['loader']['name'])
    divide_way = parse_dict(config['loader'], 'divide', 'random')
    if divide_way == "random":
        seq_datas, out_datas = loader(**config['loader']['args'])
        test_size = parse_dict(config, "test_size", 0.2)
        train_seq_datas, test_seq_datas, train_labels, test_labels = \
            train_test_split(seq_datas, out_datas,
                             test_size=test_size, random_state=SEED)
    else:
        base_dict = config['loader']['args']
        base_dict['keys'] = config['loader']['train_key']
        train_seq_datas, train_labels = loader(**base_dict)
        base_dict['keys'] = config['loader']['test_key']
        test_seq_datas, test_labels = loader(**base_dict)

    print("Training Dataset: input shape is {}".format(train_seq_datas.shape))
    print("Training Dataset: label shape is {}".format(train_labels.shape))

    generator = getattr(mot_loader, config['generator']['name'])
    train_dataset = generator(
        input_data=train_seq_datas, label_data=train_labels)
    test_dataset = generator(
        input_data=test_seq_datas, label_data=test_labels)
    return train_dataset, test_dataset

def load_da_dataset(config):
    """Load specified dataset and split it to train/val."""
    loader = getattr(mot_loader, config['loader']['name'])
    divide_way = parse_dict(config['loader'], 'divide', 'random')
    if divide_way == "random":
        frame_datas, out_datas, target_ids, imgs_path = \
            loader(**config['loader']['args'])
        test_size = parse_dict(config, "test_size", 0.2)

        (train_frame_datas, test_frame_datas, train_labels, test_labels,
            train_target_ids, test_target_ids, train_imgs_path, test_imgs_path) = \
                split_dataset_4(frame_datas, out_datas, target_ids, imgs_path,
                              test_size=test_size, random_state=SEED)
    else:
        base_dict = config['loader']['args']
        base_dict['keys'] = config['loader']['train_key']
        train_frame_datas, train_labels, train_target_ids, train_imgs_path =\
            loader(**base_dict)
        base_dict['keys'] = config['loader']['test_key']
        test_frame_datas, test_labels, test_target_ids, test_imgs_path =\
            loader(**base_dict)


    generator = getattr(mot_loader, config['generator']['name'])
    img_size = parse_size(parse_dict(config['generator']['args'], 'mode', None))
    train_dataset = generator(
        input_data=train_frame_datas, label_data=train_labels,
        target_id=train_target_ids, imgs_path=train_imgs_path,
        use_siamese=config['generator']['args']['use_siamese'], size=img_size)

    test_dataset = generator(
        input_data=train_frame_datas, label_data=train_labels,
        target_id=train_target_ids, imgs_path=train_imgs_path,
        use_siamese=config['generator']['args']['use_siamese'], size=img_size)
    return train_dataset, test_dataset

def load_cuhk_dataset(config):
    """Load specified dataset and split it to train/val."""
    loader = getattr(cuhk_loader, config['loader']['name'])
    # data_loader = load_datasets_cuhk, load_datasets_cuhk_mot
    imgs, x1_index, x2_index, labels = loader(**config['loader']['args'])
    test_size = parse_dict(config, 'test_size', 0.2)
    (train_x1_index, test_x1_index, train_x2_index, test_x2_index,
     train_labels, test_labels) = \
         split_dataset(x1_index, x2_index, labels,
                         test_size=test_size, random_state=SEED)

    print("Training Dataset: input shape is {}".format(train_x1_index.shape))
    print("Test Dataset: input shape is {}".format(test_x1_index.shape))

    generator = getattr(cuhk_loader, config['generator']['name'])
    train_dataset = generator(
        x1=train_x1_index,
        x2=train_x2_index,
        label_data=train_labels,
        imgs=imgs)

    test_dataset = generator(
        x1=test_x1_index,
        x2=test_x2_index,
        label_data=test_labels,
        imgs=imgs)
    return train_dataset, test_dataset

def load_dataset(config):
    if config['type'] == "rnn":
        return load_rnn_dataset(config)
    elif config['type'] == 'da':
        return load_da_dataset(config)
    elif config['type'] == 'cuhk':
        return load_cuhk_dataset(config)
    elif config['type'] == 'test':
        loader = getattr(mot_loader, config['loader']['name'])
        return loader(**config['loader']['args'])

def get_enet_model(config):
    Model = getattr(enet_paper, config['name'])
    return Model(config["architecture"])


def get_model(config, model_type=None):
