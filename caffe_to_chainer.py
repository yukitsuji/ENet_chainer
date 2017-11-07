#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys  # NOQA  # isort:skip
sys.path.insert(0, '.')  # NOQA  # isort:skip

import os
import re

import numpy as np
from google.protobuf import text_format

from converter import caffe_pb2
import chainer
import chainer.links as L
from chainer import serializers

# from enet.models.enet_paper
from enet.config_utils import get_model, parse_args


def get_chainer_model(n_class, input_size, n_blocks, pyramids, mid_stride):
    with chainer.using_config('train', True):
        model = pspnet.PSPNet(
            n_class, input_size, n_blocks, pyramids, mid_stride)
        model(np.random.rand(1, 3, input_size, input_size).astype(np.float32))
    size = 0
    for param in model.params():
        try:
            size += param.size
        except Exception as e:
            print(str(type(e)), e, param, param.name)
            exit(-1)
    print('PSPNet (chainer) size:', size)
    return model


def get_param_net(prodo_dir, param_fn, proto_fn):
    print('Loading caffe parameters...', end=' ')
    param = caffe_pb2.NetParameter()
    param.MergeFromString(open(param_fn, 'rb').read())
    print('done')

    print('Loading caffe prototxt...', end=' ')
    proto_fp = open(proto_fn).read()
    net = caffe_pb2.NetParameter()
    net = text_format.Merge(proto_fp, net)
    print('done')
    print(net.layer[0])
    return param, net


def copy_conv(layer, config, conv, has_bias=False):
    data = np.array(layer.blobs[0].data)
    conv.W.data[:] = data.reshape(conv.W.shape)
    if has_bias:
        data = np.array(layer.blobs[1].data)
        conv.b.data[:] = data.reshape(conv.b.shape)

    # Check ksize
    assert config.convolution_param.kernel_size[0] == conv.ksize, \
        'ksize: {} != {} ({}, {}, {}, {})'.format(
            config.convolution_param.kernel_size[0], conv.ksize,
            layer.name, config, conv, conv.name)

    # Check stride
    if len(config.convolution_param.stride) == 1:
        stride = config.convolution_param.stride[0]
        stride = (stride, stride)
    assert stride == conv.stride, \
        'stride: {} != {} ({}, {}, {}, {})'.format(
            stride, conv.stride, layer.name, config, conv, conv.name)

    # Check pad
    if len(config.convolution_param.pad) == 1:
        pad = config.convolution_param.pad[0]
        pad = (pad, pad)
    elif config.convolution_param.pad == []:
        pad = (0, 0)
    assert pad == conv.pad, \
        'pad: {} != {} ({}, {}, {}, {})'.format(
            pad, conv.pad, layer.name, config, conv, conv.name)

    assert layer.convolution_param.bias_term == has_bias
    if not has_bias:
        assert conv.b is None
    if isinstance(config.convolution_param.dilation, int):
        assert isinstance(conv, L.DilatedConvolution2D)
        assert config.convolution_param.dilation == conv.dilate

    return conv


def copy_cbr(layer, config, cbr):
    if 'Convolution' in layer.type:
        cbr.conv = copy_conv(layer, config, cbr.conv)
    elif 'BN' in layer.type:
        cbr.bn.eps = config.bn_param.eps
        cbr.bn.decay = config.bn_param.momentum
        cbr.bn.gamma.data.ravel()[:] = np.array(layer.blobs[0].data).ravel()
        cbr.bn.beta.data.ravel()[:] = np.array(layer.blobs[1].data).ravel()
        cbr.bn.avg_mean.ravel()[:] = np.array(layer.blobs[2].data).ravel()
        cbr.bn.avg_var.ravel()[:] = np.array(layer.blobs[3].data).ravel()
    else:
        print('copy cbr Ignored: {} ({})'.format(layer.name, layer.type))
    return cbr

def copy_bn(layer, config, bn):
    print(config.bn_param)
    # bn.eps = config.bn_param.eps
    # bn.decay = config.bn_param.momentum
    print(layer.blobs)
    bn.gamma.data.ravel()[:] = np.array(layer.blobs[0].data).ravel()
    bn.beta.data.ravel()[:] = np.array(layer.blobs[1].data).ravel()
    bn.avg_mean.ravel()[:] = np.array(layer.blobs[2].data).ravel()
    bn.avg_var.ravel()[:] = np.array(layer.blobs[3].data).ravel()
    return cbr


def copy_head(layer, config, block):
    if layer.name.startswith('conv0_1'):
        block.ib_conv = copy_conv(layer, config, block.ib_conv, has_bias=True)
    elif layer.name.startswith('bn0_1'):
        block.ib_bn = copy_bn(layer, config, block.ib_bn)
    elif layer.name.startswith('prelu0_1'):
        block.ib_prelu = copy_cbr(layer, config, block.ib_prelu)
    else:
        print('copy head Ignored: {} ({})'.format(layer.name, layer.type))
    return block


def copy_bottleneck(layer, config, block):
    if 'reduce' in layer.name:
        block.cbr1 = copy_cbr(layer, config, block.cbr1)
    elif '3x3' in layer.name:
        block.cbr2 = copy_cbr(layer, config, block.cbr2)
    elif 'increase' in layer.name:
        block.cbr3 = copy_cbr(layer, config, block.cbr3)
    elif 'proj' in layer.name:
        block.cbr4 = copy_cbr(layer, config, block.cbr4)
    else:
        print('bottleneck Ignored: {} ({})'.format(layer.name, layer.type))
    return block


def copy_resblock(layer, config, block):
    if '/' in layer.name:
        layer.name = layer.name.split('/')[0]
    i = int(layer.name.split('_')[1]) - 1
    block._children[i] = copy_bottleneck(layer, config, block[i])
    return block


def copy_ppm_module(layer, config, block):
    ret = re.search('pool([0-9]+)', layer.name)
    if ret is None:
        raise ValueError('Error in copy_ppm_module:'
                         '{}, {}, {}'.format(layer.name, config, block))
    i = int(ret.groups()[0])
    i = {1: 3,
         2: 2,
         3: 1,
         6: 0}[i]
    block._children[i] = copy_cbr(layer, config, block[i])
    return block


def transfer(model, param, net):
    name_config = dict([(l.name, l) for l in net.layer])
    print(model._children)
    for layer in param.layer:
        if layer.name not in name_config:
            continue
        config = name_config[layer.name]
        if layer.name.startswith('conv0') or layer.name.startswith('bn0') or layer.name.startswith('prelu0'):
            print(layer.name)
            model.initial_block_0 = copy_head(layer, config, model.initial_block_0)
        elif layer.name.startswith('conv1'):
            pass
        elif layer.name.startswith('conv2'):
            model.trunk.res2 = copy_resblock(layer, config, model.trunk.res2)
        elif layer.name.startswith('conv3'):
            model.trunk.res3 = copy_resblock(layer, config, model.trunk.res3)
        elif layer.name.startswith('conv4'):
            model.trunk.res4 = copy_resblock(layer, config, model.trunk.res4)
        elif layer.name.startswith('conv5') \
                and 'pool' not in layer.name \
                and 'conv5_4' not in layer.name:
            model.trunk.res5 = copy_resblock(layer, config, model.trunk.res5)
        elif layer.name.startswith('conv5_3') and 'pool' in layer.name:
            model.ppm = copy_ppm_module(layer, config, model.ppm)
        elif layer.name.startswith('conv5_4'):
            model.cbr_main = copy_cbr(layer, config, model.cbr_main)
        elif layer.name.startswith('conv6'):
            model.out_main = copy_conv(
                layer, config, model.out_main, has_bias=True)
        else:
            print('transfer Ignored: {} ({})'.format(layer.name, layer.type))
    return model


if __name__ == '__main__':
    proto_dir = 'converter'

    if not os.path.exists(os.path.join(proto_dir, 'weights2.caffemodel')):
        print('Please download pspnet101_cityscapes.caffemodel from here: '
              'https://drive.google.com/open?id=0BzaU285cX7TCT1M3TmNfNjlUeEU '
              'and put it into weights/ dir.')
        exit()

    # Num of parameters of models for...
    # VOC2012: 65708501 (train: 70524906)
    # Cityscapes: 65707475
    # ADE20K: 46782550

    settings = {
        'cityscapes': {
            'proto_fn': 'weights2.prototxt',
            'param_fn': 'weights2.caffemodel',
            'n_class': 19,
            'input_size': 713,
            'n_blocks': [3, 4, 23, 3],
            'feat_size': 90,
            'mid_stride': True,
            'pyramids': [6, 3, 2, 1],
        },
    }

    for dataset_name in ['cityscapes']:
        proto_fn = settings[dataset_name]['proto_fn']
        param_fn = settings[dataset_name]['param_fn']
        n_class = settings[dataset_name]['n_class']
        input_size = settings[dataset_name]['input_size']
        n_blocks = settings[dataset_name]['n_blocks']
        pyramids = settings[dataset_name]['pyramids']
        mid_stride = settings[dataset_name]['mid_stride']

        name = os.path.splitext(proto_fn)[0]
        param_fn = os.path.join(proto_dir, param_fn)
        proto_fn = os.path.join(proto_dir, proto_fn)

        config = parse_args()
        model = get_model(config['model'])
        param, net = get_param_net(proto_dir, param_fn, proto_fn)
        model = transfer(model, param, net)

        serializers.save_npz(
            'weights/{}_reference.npz'.format(name), model)
        print('weights/{}_reference.npz'.format(name), 'saved')
