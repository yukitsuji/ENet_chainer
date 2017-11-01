#!/usr/env/bin python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import sys
import subprocess
import os
import yaml

import chainer
from chainer import cuda, optimizers, serializers
from chainer import training

from enet.config_utils import *

chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"


from collections import OrderedDict
yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))

from models import enet_paper

def train_enet():
    """Training ENet."""
    config = parse_args()
    rnn_predictor = get_model(config["model"])
    train_data, test_data = load_dataset(config["dataset"])
    train_iter, test_iter = create_iterator(train_data, test_data, config['iterator'])
    optimizer = create_optimizer(config['optimizer'], rnn_predictor)
    args = {}
    updater = create_updater(train_iter, optimizer, config['updater'], args)
    trainer = training.Trainer(updater, config['end_trigger'], out=config['results'])
    trainer = create_extension(trainer, test_iter,  rnn_predictor, config['extension'])
    trainer.run()
    chainer.serializers.save_npz(os.path.join(config['results'], 'rnn_predictor.npz'),
                                 rnn_predictor)

def main():
    train_enet()

if __name__ == '__main__':
    main()
