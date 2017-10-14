#!/usr/env/bin python3
import argparse
import yaml

from collections import OrderedDict
yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))

from models import enet_paper

def get_trainer(args):
    config = yaml.load(file(args.config))
    model_config = config["model"]
    data_config = config["dataset"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ENet')
    parser.add_argument('--config', type=str)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--result_dir', type=str, default="./results")
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    trainer = get_trainer(args)
    trainer.run()
