from __future__ import division

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainercv.transforms import resize
from chainercv.utils import download_model
from spatial_dropout import spatial_dropout

class ConvBN(chainer.Chain):
    """Convolution2D + Batch Normalization"""
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False):
        super(ConvBN, self).__init__()
        with self.init_scope():
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
                 nobias=False):
        super(ConvBNPReLU, self).__init__(in_ch, out_ch, ksize, stride, pad,
                                          dilation, nobias)
        with self.init_scope():
            self.prelu = L.PReLU()

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
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False, symmetric=False):
        super(InitialBlock, self).__init__()
        with self.init_scope():
            self.ib_conv = L.Convolution2D(in_ch, out_ch, ksize, stride,
                                           pad, dilation, nobias=nobias)
            self.ib_bn = L.BatchNormalization(out_ch, eps=1e-5, decay=0.95)
            self.ib_prelu = L.PReLU()

    def __call__(self, x):
        h1 = self.ib_conv(x)
        h2 = F.max_pooling_2d(x, 2, 2)
        h = F.concat((h1, h2), axis=1)
        h = self.ib_bn(h)
        return self.ib_prelu(h)


class Block(chainer.Chain):
    """Block Abstract"""
    def __init__(self, in_ch, mid_ch, out_ch, ksize=0, stride=0, pad=0,
                 dilation=1, drop_ratio=0.1, downsample=False, nobias=False,
                 symmetric=False):
        super(Bottleneck1, self).__init__()
        self.drop_ratio = drop_ratio
        self.downsample = downsample
        with self.init_scope():
            k1, k2, s1 = self.calc_param(downsample, symmetric)
            self.block1 = ConvBNPReLU(in_ch, mid_ch, k1, s1, 0, nobias=True)
            Conv_Block = SymmetricConvBNPReLU if symmetric else ConvBNPReLU
            self.block2 = Conv_Block(mid_ch, mid_ch, k2, 1, dilation, dilation,
                                     symmetric=symmetric, nobias=False)
            self.block3 = ConvBN(mid_ch, out_ch, 1, 1, 0, nobias=True)
            self.prelu = L.PReLU()
            if downsample:
                self.conv = L.Convolution2D(in_ch, out_ch, 1, 1, 0, nobias=True)
                self.bn = L.BatchNormalization(out_ch, eps=1e-5, decay=0.95)

    def calc_param(self, downsample, symmetric):
        k1, s1 = (2, 2) if downsample else (1, 1)
        k2 = 5 if symmetric else 3
        return k1, k2, s1

    def __call__(self, x):
        h1 = self.block1(x)
        h1 = self.block2(h1)
        h1 = self.block3(h1)
        h1 = spatial_dropout(h1, drop_ratio)
        h1 = h1 if not self.downsample else h1 + self.bn(self.conv(x))
        return self.prelu(h1)

    def inference(self, x):
        h1 = self.block1(x)
        h1 = self.block2(h1)
        h1 = self.block3(h1)
        h1 = h1 if not self.downsample else h1 + self.bn(self.conv(x))
        return self.prelu(h1)

class Bottleneck1(chainer.Chain):
    """Bottleneck1"""
    def __init__(self, config, drop_ratio=0.1):
        super(Bottleneck1, self).__init__()
        self.drop_ratio = drop_ratio
        config = config["bottle1"]
        config1 = config["block1"]
        config2 = config["block2"]
        config3 = config["block3"]
        config4 = config["block4"]
        with self.init_scope():
            self.block1 = Block(config1)
            self.block2 = Block(config2)
            self.block3 = Block(config3)
            self.block4 = Block(config4)

    def __call__(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

class Bottleneck2(chainer.Chain):

    def __init__(self):
        pass

    def __call__(self):
        pass

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

class ENetBasic(chainer.Chain):
    """ENet Basic for semantic segmentation."""
    _models = {
        'camvid': {
            'n_class': 11,
        }
        'cityscapes' : {
            'n_class' : 19,
        }
    }

    def __init__(self, n_class=None, pretrained_model=None, initialW=None,
                 model_config=None, remove_dec=False):
        if n_class is None:
            if pretrained_model not in self._models:
                raise ValueError(
                    'The n_class needs to be supplied as an argument.')
            n_class = self._models[pretrained_model]['n_class']

        if model_config is None:
            raise ValueError(
                'The model config needs to be supplied as an argument.')
        else:
            config1 = model_config['bottle1']
            config2 = model_config['bottle2']
            config3 = model_config['bottle3']
            if not remove_dec:
                config4 = model_config['bottle4']
                config5 = model_config['bottle5']

        if initialW is None:
            initialW = chainer.initializers.HeNormal()

        super(ENetBasic, self).__init__()
        with self.init_scope():
            self.bottle1 = Bottleneck1()
            self.bottle2 = Bottleneck2()
            self.bottle3 = Bottleneck2()
            if remove_dec:
                self.deconv5
            else:
                self.bottle4 = Bottleneck4()
                self.bottle5 = Bottleneck5()
                self.deconv6 = L.DilatedConvolution2D()
            # self.conv_classifier = L.Convolution2D(
            #     64, n_class, 1, 1, 0, initialW=initialW)

        self.n_class = n_class

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
