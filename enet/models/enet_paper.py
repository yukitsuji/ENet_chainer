from __future__ import division

import numpy as np
import sys

import chainer
import chainer.functions as F
import chainer.links as L
# from chainer import chain

from chainercv.transforms import resize
from chainercv.utils import download_model
from enet.models.spatial_dropout import spatial_dropout

from chainer import Variable
from chainercv.links import PixelwiseSoftmaxClassifier


def parse_dict(dic, key, value=None):
    return value if not key in dic else dic[key]

def _without_cudnn(f, x):
    with chainer.using_config('use_cudnn', 'never'):
        return f.apply((x,))[0]


class Conv(chainer.Chain):
    "Convolution2D for inference module"
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False, upsample=False):
        super(Conv, self).__init__()
        with self.init_scope():
            if upsample:
                self.conv = L.Deconvolution2D(
                                in_ch, out_ch, ksize, stride, pad, nobias=nobias)
            else:
                if dilation > 1:
                    self.conv = L.DilatedConvolution2D(
                        in_ch, out_ch, ksize, stride, pad, dilation, nobias=nobias)
                else:
                    self.conv = L.Convolution2D(
                        in_ch, out_ch, ksize, stride, pad, nobias=nobias)

    def __call__(self, x):
        return self.conv(x)

    def predict(self, x):
        return self.conv(x)


class ConvBN(Conv):
    """Convolution2D + Batch Normalization"""
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False, upsample=False):
        super(ConvBN, self).__init__(in_ch, out_ch, ksize, stride, pad,
                                     dilation, nobias, upsample)

        self.add_link("bn", L.BatchNormalization(out_ch, eps=1e-5, decay=0.95))

    def __call__(self, x):
        return self.bn(self.conv(x))

    def predict(self, x):
        return self.bn(self.conv(x))


class ConvPReLU(Conv):
    """Convolution2D + PReLU"""
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False, upsample=False):
        super(ConvPReLU, self).__init__(in_ch, out_ch, ksize, stride, pad,
                                        dilation, nobias, upsample)

        self.add_link("prelu", L.PReLU())

    def __call__(self, x):
        return self.prelu(self.conv(x))

    def predict(self, x):
        return self.prelu(self.conv(x))


class ConvBNPReLU(ConvBN):
    """Convolution2D + Batch Normalization + PReLU"""
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False, upsample=False):
        super(ConvBNPReLU, self).__init__(in_ch, out_ch, ksize, stride, pad,
                                          dilation, nobias, upsample)

        self.add_link("prelu", L.PReLU())

    def __call__(self, x):
        return self.prelu(self.bn(self.conv(x)))

    def predict(self, x):
        return self.prelu(self.bn(self.conv(x)))


class SymmetricConvPReLU(chainer.Chain):
    """Convolution2D + PReLU"""
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False, upsample=None):
        super(SymmetricConvPReLU, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_ch, out_ch, (ksize, 1), stride, pad, nobias=nobias)
            self.conv2 = L.Convolution2D(
                in_ch, out_ch, (1, ksize), stride, pad, nobias=nobias)
            self.prelu = L.PReLU()

    def __call__(self, x):
        return self.prelu(self.conv2(self.conv1(x)))

    def predict(self, x):
        return self.prelu(self.conv2(self.conv1(x)))


class SymmetricConvBNPReLU(SymmetricConvPReLU):
    """Convolution2D + Batch Normalization + PReLU"""
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False, upsample=None):
        super(SymmetricConvBNPReLU, self).__init__(in_ch, out_ch, ksize,
                                                   stride, pad, dilation,
                                                   nobias, upsample)
        self.add_link("bn", L.BatchNormalization(out_ch, eps=1e-5, decay=0.95))

    def __call__(self, x):
        h = self.conv2(self.conv1(x))
        return self.prelu(self.bn(h))

    def predict(self, x):
        h = self.conv2(self.conv1(x))
        return self.prelu(self.bn(h))


class InitialBlock(chainer.Chain):
    """Initial Block"""
    def __init__(self, in_ch=3, out_ch=13, ksize=3, stride=2, pad=1,
                 nobias=False, use_bn=True):
        super(InitialBlock, self).__init__()
        with self.init_scope():
            self.ib_conv = L.Convolution2D(in_ch, out_ch, ksize, stride,
                                           pad=pad, nobias=nobias)
            self.ib_bn = L.BatchNormalization(out_ch + in_ch, eps=1e-5, decay=0.95)
            self.ib_prelu = L.PReLU()

    def __call__(self, x):
        h1 = self.ib_conv(x)
        h2 = F.max_pooling_2d(x, 2, 2)
        h = F.concat((h1, h2), axis=1)
        h = self.ib_bn(h)
        return self.ib_prelu(h)

    def predict(self, x):
        h1 = self.ib_conv(x)
        h2 = F.max_pooling_2d(x, 2, 2)
        h = F.concat((h1, h2), axis=1)
        h = self.ib_bn(h)
        return self.ib_prelu(h)


class Block(chainer.Chain):
    """Block Abstract"""
    def __init__(self, in_ch=3, mid_ch=0, out_ch=13, ksize=3, stride=1, pad=1,
                 dilation=1, nobias=False, symmetric=False, drop_ratio=0.0,
                 downsample=False, upsample=False, p=None, use_bn=True):
        super(Block, self).__init__()
        k1, k2, s1 = self.calc_param(downsample, symmetric)
        self.p = p
        self.drop_ratio = drop_ratio
        self.downsample = downsample
        self.upsample = upsample

        with self.init_scope():
            this_mod = sys.modules[__name__]
            conv_type = "ConvBN" if use_bn else "Conv"
            ConvBlock = getattr(this_mod, conv_type + "PReLU")
            self.conv1 = ConvBlock(in_ch, mid_ch, k1, s1, 0, nobias=True)

            conv_type2 = conv_type + "PReLU"
            conv_type2 = "Symmetric" + conv_type2 if symmetric else conv_type2
            ConvBlock = getattr(this_mod, conv_type2)
            self.conv2 = ConvBlock(mid_ch, mid_ch, k2, stride,
                                   pad, dilation,
                                   nobias=False,
                                   upsample=upsample)

            ConvBlock = getattr(this_mod, conv_type)
            self.conv3 = ConvBlock(mid_ch, out_ch, 1, 1, 0, nobias=True)
            self.prelu = L.PReLU()
            if downsample:
                ConvBlock = getattr(this_mod, conv_type)
                self.conv = ConvBlock(in_ch, out_ch, 1, 1, 0, nobias=True)
            # if self.upsample:
                # self.p = p

    def calc_param(self, downsample, symmetric):
        k1, s1 = (2, 2) if downsample else (1, 1)
        k2 = 5 if symmetric else 3
        return k1, k2, s1

    def _upsampling_2d(self, x, pool):
        if x.shape != pool.indexes.shape:
            min_h = min(x.shape[2], pool.indexes.shape[2])
            min_w = min(x.shape[3], pool.indexes.shape[3])
            x = x[:, :, :min_h, :min_w]
            pool.indexes = pool.indexes[:, :, :min_h, :min_w]
        outsize = (x.shape[2] * 2, x.shape[3] * 2)
        return F.upsampling_2d(
            x, pool.indexes, ksize=(pool.kh, pool.kw),
            stride=(pool.sy, pool.sx), pad=(pool.ph, pool.pw), outsize=outsize)

    def __call__(self, x):
        h1 = self.conv1(x)
        h1 = self.conv2(h1)
        h1 = self.conv3(h1)
        h1 = spatial_dropout(h1, self.drop_ratio)
        if self.downsample:
            self.p = F.MaxPooling2D(2, 2)
            h1 += self.conv(_without_cudnn(self.p, x))
        elif self.upsample:
            h1 += self._upsampling_2d(self.conv(self.bn(x)), self.p)
        else:
            h1 += x
        # h1 = h1 if not self.downsample else h1 + self.bn(self.conv(x))
        return self.prelu(h1)

    def predict(self, x):
        h1 = self.conv1(x)
        h1 = self.conv2(h1)
        h1 = self.conv3(h1)
        if self.downsample:
            self.p = F.MaxPooling2D(2, 2)
            h1 += self.conv(_without_cudnn(self.p, x))
        elif self.upsample:
            h1 += self._upsampling_2d(self.conv(self.bn(x)), self.p)
        else:
            h1 += x
        return self.prelu(h1)


class Bottleneck2(chainer.Chain):
    """Bottleneck1"""
    def __init__(self, in_ch=3, mid_ch=0, out_ch=13, drop_ratio=0.0,
                 use_bn=True):
        super(Bottleneck2, self).__init__()
        basic_config = {"in_ch": in_ch, "mid_ch": mid_ch, "out_ch": out_ch,
                        "drop_ratio": drop_ratio, "use_bn": use_bn}
        with self.init_scope():
            self.block1 = Block(**basic_config)
            self.block2 = Block(**self.to_dilated(basic_config, 2))
            self.block3 = Block(**self.to_symmetric(basic_config))
            self.block4 = Block(**self.to_dilated(basic_config, 4))
            self.block5 = Block(**basic_config)
            self.block6 = Block(**self.to_dilated(basic_config, 8))
            self.block7 = Block(**self.to_symmetric(basic_config))
            self.block8 = Block(**self.to_dilated(basic_config, 16))

    def to_dilated(self, basic_config, dilation):
        config = basic_config.copy()
        config.update({"dilation": dilation, "pad": dilation})
        return config

    def to_symmetric(self, basic_config):
        config = basic_config.copy()
        config.update({"symmetric": True})
        return config

    def __call__(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        return x

    def predict(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        return x


class FullConv(chainer.Chain):
    """FullConv Abstract"""
    def __init__(self, in_ch=3, mid_ch=0, out_ch=13, ksize=3, stride=1, pad=1):
        super(FullConv, self).__init__()
        with self.init_scope():
            self.deconv = L.Deconvolution2D(in_ch, out_ch, ksize, stride, pad)

    def __call__(self, x):
        return self.deconv(x)

    def predict(self, x):
        return self.deconv(x)


class ENetBasic(chainer.Chain):
    """ENet Basic for semantic segmentation."""
    def __init__(self, model_config, pretrained_model=None):
        super(ENetBasic, self).__init__()
        n_class = None
        this_mod = sys.modules[__name__]
        self.layers = []
        with self.init_scope():
            for key, config in model_config.items():
                Model = getattr(this_mod, config['type'])
                loop = parse_dict(config, 'loop', 1)
                for l in range(loop):
                    layer = Model(**config['args'])
                    name = key + '_{}'.format(l)
                    setattr(self, name, layer)
                    self.layers.append(getattr(self, name))

        if pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, x):
        with chainer.using_config('train', False), \
                chainer.function.no_backprop_mode():
            x = self.xp.asarray(x)
            if x.ndim == 3:
                x = self.xp.expand_dims(x, 0)
            for layer in self.layers:
                x = layer.predict(x)
            label = self.xp.argmax(x.data, axis=1).astype("i")
            label = chainer.cuda.to_cpu(label)
            return list(label)

    #     h = self._upsampling_2d(h, p4)
    #     h = self.conv_decode4_bn(self.conv_decode4(h))
    #     h = self._upsampling_2d(h, p3)
    #     h = self.conv_decode3_bn(self.conv_decode3(h))
    #     h = self._upsampling_2d(h, p2)
    #     h = self.conv_decode2_bn(self.conv_decode2(h))
    #     h = self._upsampling_2d(h, p1)
    #     h = self.conv_decode1_bn(self.conv_decode1(h))
    #     score = self.conv_classifier(h)
    #     return score
