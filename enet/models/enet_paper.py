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


def _without_cudnn(f, x):
    with chainer.using_config('use_cudnn', 'never'):
        return f(x)

class ConvBN(chainer.Chain):
    """Convolution2D + Batch Normalization"""
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False, upsample=False):
        super(ConvBN, self).__init__()
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

            self.bn = L.BatchNormalization(out_ch, eps=1e-5, decay=0.95)

    def __call__(self, x):
        return self.bn(self.conv(x))


class ConvBNPReLU(ConvBN):
    """Convolution2D + Batch Normalization + PReLU"""
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False, upsample=False):
        super(ConvBNPReLU, self).__init__(in_ch, out_ch, ksize, stride, pad,
                                          dilation, nobias, upsample)

        self.add_link("prelu", L.PReLU())
        # with self.init_scope():
        #     self.prelu = L.PReLU()

    def __call__(self, x):
        return self.prelu(self.bn(self.conv(x)))


# class ConvBNPReLU(chainer.Chain):
#     """Convolution2D + Batch Normalization + PReLU"""
#     def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
#                  nobias=False):
#         super(ConvBNPReLU, self).__init__()
#         with self.init_scope():
#             if dilation > 1:
#                 self.conv = L.DilatedConvolution2D(
#                     in_ch, out_ch, ksize, stride, pad, dilation, nobias=nobias)
#             else:
#                 self.conv = L.Convolution2D(
#                     in_ch, out_ch, ksize, stride, pad, nobias=nobias)
#
#             self.bn = L.BatchNormalization(out_ch, eps=1e-5, decay=0.95)
#             self.prelu = L.PReLU()
#
#     def __call__(self, x):
#         return self.prelu(self.bn(self.conv(x)))


class SymmetricConvBNPReLU(chainer.Chain):
    """Convolution2D + Batch Normalization + PReLU"""
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False):
        super(SymmetricConvBNPReLU, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_ch, out_ch, (ksize, 1), stride, pad, nobias=nobias)
            self.conv2 = L.Convolution2D(
                in_ch, out_ch, (1, ksize), stride, pad, nobias=nobias)
            self.bn = L.BatchNormalization(out_ch, eps=1e-5, decay=0.95)
            self.prelu = L.PReLU()

    def __call__(self, x):
        h = self.conv2(self.conv1(x))
        return self.prelu(self.bn(h))


class InitialBlock(chainer.Chain):
    """Initial Block"""
    def __init__(self, in_ch=3, out_ch=13, ksize=3, stride=2, pad=1, dilation=1,
                 nobias=False, symmetric=False):
        super(InitialBlock, self).__init__()
        with self.init_scope():
            self.ib_conv = L.Convolution2D(in_ch, out_ch, ksize, stride,
                                           pad=pad, nobias=nobias)
            self.ib_bn = L.BatchNormalization(out_ch, eps=1e-5, decay=0.95)
            self.ib_prelu = L.PReLU()

    def __call__(self, x):
        h1 = self.ib_conv(x)
        h2 = F.max_pooling_2d(x, 2, 2)
        h = F.concat((h1, h2), axis=1)
        h = self.ib_bn(h)
        return self.ib_prelu(h)

    def inference(self, x):
        h1 = self.ib_conv(x)
        h2 = F.max_pooling_2d(x, 2, 2)
        h = F.concat((h1, h2), axis=1)
        h = self.ib_bn(h)
        return self.ib_prelu(h)


# class Block(chainer.Chain):
#     """Block Abstract"""
#     def __init__(self, in_ch, mid_ch, out_ch, ksize=0, stride=0, pad=0,
#                  dilation=1, drop_ratio=0.1, downsample=False, nobias=False,
#                  symmetric=False):
#         super(Bottleneck1, self).__init__()
#         self.drop_ratio = drop_ratio
#         self.downsample = downsample
#         with self.init_scope():
#             k1, k2, s1 = self.calc_param(downsample, symmetric)
#             self.block1 = ConvBNPReLU(in_ch, mid_ch, k1, s1, 0, nobias=True)
#             Conv_Block = SymmetricConvBNPReLU if symmetric else ConvBNPReLU
#             self.block2 = Conv_Block(mid_ch, mid_ch, k2, 1, dilation, dilation,
#                                      symmetric=symmetric, nobias=False)
#             self.block3 = ConvBN(mid_ch, out_ch, 1, 1, 0, nobias=True)
#             self.prelu = L.PReLU()
#             if downsample:
#                 self.conv = L.Convolution2D(in_ch, out_ch, 1, 1, 0, nobias=True)
#                 self.bn = L.BatchNormalization(out_ch, eps=1e-5, decay=0.95)

class Block(chainer.Chain):
    """Block Abstract"""
    def __init__(self, config):
        super(Block, self).__init__()
        self.initialize_param()
        self.parse_config(config)
        k1, k2, s1 = self.calc_param()

        with self.init_scope():
            self.block1 = ConvBNPReLU(self.in_ch, self.mid_ch, k1, s1, 0,
                                      nobias=True)
            ConvBlock = SymmetricConvBNPReLU if self.symmetric else ConvBNPReLU
            self.block2 = ConvBlock(self.mid_ch, self.mid_ch, k2, 1,
                                    self.dilation, self.dilation,
                                    nobias=False, symmetric=self.symmetric,
                                    upsample=self.upsample)
            self.block3 = ConvBN(self.mid_ch, self.out_ch, 1, 1, 0, nobias=True)
            self.prelu = L.PReLU()
            if self.downsample:
                self.conv = L.Convolution2D(self.in_ch, self.out_ch, 1, 1, 0, nobias=True)
                self.bn = L.BatchNormalization(self.out_ch, eps=1e-5, decay=0.95)
            # if self.upsample:
                # self.p =

    def initialize_param(config):
        self.in_ch = 0
        self.mid_ch = 0
        self.out_ch = 0
        self.ksize = 1
        self.stride = 0
        self.pad = 0
        self.dilation = 1
        self.drop_ratio = 0.1
        self.downsample = False
        self.upsample = False
        self.nobias = False
        self.symmetric = False
        self.p = False

    def parse_config(config):
        for key, value in config.items():
            setattr(self, key, value)

    def calc_param(self):
        k1, s1 = (2, 2) if self.downsample else (1, 1)
        k2 = 5 if self.symmetric else 3
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
        h1 = self.block1(x)
        h1 = self.block2(h1)
        h1 = self.block3(h1)
        h1 = spatial_dropout(h1, drop_ratio)
        if self.downsample:
            p = F.MaxPooling2D(2, 2)
            h1 += self.bn(self.conv(_without_cudnn(p, x)))
        elif self.upsample:
            h1 += self._upsampling_2d(self.conv(self.bn(x)), self.p)
        # h1 = h1 if not self.downsample else h1 + self.bn(self.conv(x))
        return self.prelu(h1)

    def inference(self, x):
        h1 = self.block1(x)
        h1 = self.block2(h1)
        h1 = self.block3(h1)
        h1 = h1 if not self.downsample else h1 + self.bn(self.conv(x))
        return self.prelu(h1)

class FullConv(chainer.Chain):
    """Last Layer"""
    def __init__(self):
        pass

    def __call__(self):
        pass


class Architecture(chainer.Chain):
    """Architecture Abstract"""
    def __init__(self, config):
        super(Bottleneck, self).__init__()
        model_config = config["architecture"]
        self.layers = []
        for key in model_config.keys():
            c = model_config[key]
            Model = self.get_model(c["type"])
            num_loop = 1 if not "loop" in c.keys() else c["loop"]
            for i in range(num_loop):
                name = key + "_{}".format(i + 1)
                self.add_link(name, Model(c["args"]))
                self.layers.append(getattr(self, name))

    def get_model(self, model_type):
        if model_type == "initial":
            return InitialBlock
        elif model_type == "block":
            return Block
        elif model_type == "bottleneck1":
            return Bottleneck1
        elif model_type == "bottleneck2":
            return Bottleneck2
        elif model_type == "bottleneck3":
            return Bottleneck3
        elif model_type == "bottleneck4":
            return Bottleneck4
        elif model_type == "bottleneck5":
            return Bottlenec5

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Bottleneck1(chainer.Chain):
    """Bottleneck1"""
    def __init__(self, config):
        super(Bottleneck1, self).__init__()
        config = config["bottle1"]
        config1 = config["block1"]
        config2 = config["block2"]
        config3 = config["block3"]
        config4 = config["block4"]
        config5 = config["block5"]
        with self.init_scope():
            self.block1 = Block(config1)
            self.block2 = Block(config2)
            self.block3 = Block(config3)
            self.block4 = Block(config4)
            self.block5 = Block(config5)

    def __call__(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

class Bottleneck2(chainer.Chain):

    def __init__(self):
        super(Bottleneck2, self).__init__()
        config = config["bottle2"]
        config1 = config["block1"]
        config2 = config["block2"]
        config3 = config["block3"]
        config4 = config["block4"]
        config5 = config["block5"]
        with self.init_scope():
            self.block1 = Block(config1)
            self.block2 = Block(config2)
            self.block3 = Block(config3)
            self.block4 = Block(config4)
            self.block5 = Block(config5)

    def __call__(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

class Bottleneck4(chainer.Chain):

    def __init__(self):
        pass

    def __call__(self):
        pass

class Bottleneck5(chainer.Chain):

    def __init__(self):
        pass

    def __call__(self):
        pass


def parse_dict(dic, key, value=None):
    return value if not key in dic else dic[key]

class ENetBasic(chainer.Chain):
    """ENet Basic for semantic segmentation."""
    def __init__(self, model_config):
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

        # self.n_class = n_class

        if pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)

    def encoder(self, x):
        h = self.bottle1(x)
        h = self.bottle2(h)
        h = self.bottle3(h)
        return h

    def decoder(self, x):
        h = self.bottle4(x)
        h = self.bottle5(x)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

    def __call__(self, x):
        """Compute an image-wise score from a batch of images
        Args:
            x (chainer.Variable): A variable with 4D image array.
        Returns:
            chainer.Variable:
            An image-wise score. Its channel size is :obj:`self.n_class`.
        """
        p1 = F.MaxPooling2D(2, 2)
        p2 = F.MaxPooling2D(2, 2)
        p3 = F.MaxPooling2D(2, 2)
        p4 = F.MaxPooling2D(2, 2)
        h = F.local_response_normalization(x, 5, 1, 1e-4 / 5., 0.75)
        h = _without_cudnn(p1, F.relu(self.conv1_bn(self.conv1(h))))
        h = _without_cudnn(p2, F.relu(self.conv2_bn(self.conv2(h))))
        h = _without_cudnn(p3, F.relu(self.conv3_bn(self.conv3(h))))
        h = _without_cudnn(p4, F.relu(self.conv4_bn(self.conv4(h))))
        # h = self._upsampling_2d(h, p4)
        h = self.conv_decode4_bn(self.conv_decode4(h))
        # h = self._upsampling_2d(h, p3)
        h = self.conv_decode3_bn(self.conv_decode3(h))
        # h = self._upsampling_2d(h, p2)
        h = self.conv_decode2_bn(self.conv_decode2(h))
        # h = self._upsampling_2d(h, p1)
        h = self.conv_decode1_bn(self.conv_decode1(h))
        score = self.conv_classifier(h)
        return score

    def predict(self, imgs):
        """Conduct semantic segmentations from images.
        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their values are :math:`[0, 255]`.
        Returns:
            list of numpy.ndarray:
            List of integer labels predicted from each image in the input \
            list.
        """
        labels = list()
        for img in imgs:
            C, H, W = img.shape
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                x = chainer.Variable(self.xp.asarray(img[np.newaxis]))
                score = self.__call__(x)[0].data
            score = chainer.cuda.to_cpu(score)
            if score.shape != (C, H, W):
                dtype = score.dtype
                score = resize(score, (H, W)).astype(dtype)

            label = np.argmax(score, axis=0).astype(np.int32)
            labels.append(label)
        return labels
