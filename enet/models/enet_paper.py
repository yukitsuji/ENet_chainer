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


def _without_cudnn(f, x):
    with chainer.using_config('use_cudnn', 'never'):
        return f.apply((x,))[0]

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


    def __call__(self, x):
        return self.prelu(self.bn(self.conv(x)))


class SymmetricConvBNPReLU(chainer.Chain):
    """Convolution2D + Batch Normalization + PReLU"""
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False, upsample=None):
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
    def __init__(self, in_ch=3, out_ch=13, ksize=3, stride=2, pad=1,
                 nobias=False):
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

    def inference(self, x):
        h1 = self.ib_conv(x)
        h2 = F.max_pooling_2d(x, 2, 2)
        h = F.concat((h1, h2), axis=1)
        h = self.ib_bn(h)
        return self.ib_prelu(h)

class Block(chainer.Chain):
    """Block Abstract"""
    def __init__(self, in_ch=3, mid_ch=0, out_ch=13, ksize=3, stride=1, pad=1,
                 dilation=1, nobias=False, symmetric=False, drop_ratio=0.0,
                 downsample=False, upsample=False, p=None):
        super(Block, self).__init__()
        k1, k2, s1 = self.calc_param(downsample, symmetric)
        self.p = p
        self.drop_ratio = drop_ratio
        self.downsample = downsample
        self.upsample = upsample
        with self.init_scope():
            self.block1 = ConvBNPReLU(in_ch, mid_ch, k1, s1, 0,
                                      nobias=True)
            ConvBlock = SymmetricConvBNPReLU if symmetric else ConvBNPReLU
            self.block2 = ConvBlock(mid_ch, mid_ch, k2, stride,
                                    pad, dilation,
                                    nobias=False,
                                    upsample=upsample)
            self.block3 = ConvBN(mid_ch, out_ch, 1, 1, 0, nobias=True)
            self.prelu = L.PReLU()
            if downsample:
                self.conv = L.Convolution2D(in_ch, out_ch, 1, 1, 0, nobias=True)
                self.bn = L.BatchNormalization(out_ch, eps=1e-5, decay=0.95)
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
        h1 = self.block1(x)
        h1 = self.block2(h1)
        h1 = self.block3(h1)
        h1 = spatial_dropout(h1, self.drop_ratio)

        if self.downsample:
            self.p = F.MaxPooling2D(2, 2)
            h1 += self.bn(self.conv(_without_cudnn(self.p, x)))
        elif self.upsample:
            h1 += self._upsampling_2d(self.conv(self.bn(x)), self.p)
        else:
            h1 += x
        # h1 = h1 if not self.downsample else h1 + self.bn(self.conv(x))
        return self.prelu(h1)

    def inference(self, x):
        h1 = self.block1(x)
        h1 = self.block2(h1)
        h1 = self.block3(h1)
        if self.downsample:
            self.p = F.MaxPooling2D(2, 2)
            h1 += self.bn(self.conv(_without_cudnn(self.p, x)))
        elif self.upsample:
            h1 += self._upsampling_2d(self.conv(self.bn(x)), self.p)
        else:
            h1 += x
        return self.prelu(h1)


class Bottleneck2(chainer.Chain):
    """Bottleneck1"""
    def __init__(self, in_ch=3, mid_ch=0, out_ch=13, drop_ratio=0.0):
        super(Bottleneck2, self).__init__()
        basic_config = {"in_ch": in_ch, "mid_ch": mid_ch, "out_ch": out_ch,
                        "drop_ratio": drop_ratio}
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


class FullConv(chainer.Chain):
    """FullConv Abstract"""
    def __init__(self, in_ch=3, mid_ch=0, out_ch=13, ksize=3, stride=1, pad=1):
        super(FullConv, self).__init__()
        with self.init_scope():
            self.deconv = L.Deconvolution2D(in_ch, out_ch, ksize, stride, pad)

    def __call__(self, x):
        return self.deconv(x)

def parse_dict(dic, key, value=None):
    return value if not key in dic else dic[key]

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

    def __call__(self, x, y):
        for layer in self.layers:
            x = layer(x)
        return Variable(np.array([0]))

    # def __call__(self, x):
    #     """Compute an image-wise score from a batch of images
    #     Args:
    #         x (chainer.Variable): A variable with 4D image array.
    #     Returns:
    #         chainer.Variable:
    #         An image-wise score. Its channel size is :obj:`self.n_class`.
    #     """
    #     p1 = F.MaxPooling2D(2, 2)
    #     p2 = F.MaxPooling2D(2, 2)
    #     p3 = F.MaxPooling2D(2, 2)
    #     p4 = F.MaxPooling2D(2, 2)
    #     h = _without_cudnn(p1, F.relu(self.conv1_bn(self.conv1(h))))
    #     h = _without_cudnn(p2, F.relu(self.conv2_bn(self.conv2(h))))
    #     h = _without_cudnn(p3, F.relu(self.conv3_bn(self.conv3(h))))
    #     h = _without_cudnn(p4, F.relu(self.conv4_bn(self.conv4(h))))
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
