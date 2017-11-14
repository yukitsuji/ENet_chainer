from functools import partial

import cv2 as cv
import numpy as np
from PIL import Image

from chainer import datasets
from chainercv import transforms
from enet.data_util.cityscapes.cityscapes_semantic_segmentation_dataset import \
    CityscapesSemanticSegmentationDataset


def _transform(inputs, mean=None, img_size=(512, 1024), scale_label=1):
    img, label = inputs
    # Scaling
    if img_size:
        img_size = (img_size[0], img_size[1])
        img = transforms.resize(img, img_size, Image.BICUBIC)
        label = transforms.resize(
            label[None, ...], img_size, Image.NEAREST)[0]

    # Mean subtraction
    if mean is not None:
        img -= mean[:, None, None]

    if scale_label != 1:
        scale_label = (int(label.shape[1]/scale_label),\
                          int(label.shape[0]/scale_label))
        label = cv.resize(label, scale_label, interpolation=cv.INTER_NEAREST)
    return img, label


class CityscapesValidationDataset(datasets.TransformDataset):

    # Cityscapes mean
    MEAN = np.array([73.15835921, 82.90891754, 72.39239876])

    def __init__(self, data_dir="./", label_resolution="gtFine",
                 split="train", ignore_labels=True,
                 img_size=(512, 1024), scale_label=1):
        self.d = CityscapesSemanticSegmentationDataset(
            data_dir, label_resolution, split)
        t = partial(
            _transform, mean=self.MEAN, img_size=img_size,
            scale_label=scale_label)
        super().__init__(self.d, t)
