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

from enet.models import enet_paper

def train_enet():
    """Training ENet."""
    # chainer.config.enable_backprop = False
    # chainer.config.train = False
    config = parse_args()
    train_data, test_data = load_dataset(config["dataset"])
    train_iter, test_iter = create_iterator(train_data, test_data, config['iterator'])
    model = get_model(config["model"])
    optimizer = create_optimizer(config['optimizer'], model)
    devices = parse_devices(config['gpus'])
    updater = create_updater(train_iter, optimizer, config['updater'], devices)
    trainer = training.Trainer(updater, config['end_trigger'], out=config['results'])
    trainer = create_extension(trainer, test_iter,  model, config['extension'])
    trainer.run()
    chainer.serializers.save_npz(os.path.join(config['results'], 'model.npz'),
                                 model)

def main():
    train_enet()

if __name__ == '__main__':
    main()
