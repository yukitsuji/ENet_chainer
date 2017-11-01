import glob
import os

import numpy as np
from chainer import dataset
from chainer.dataset import download

from chainercv.utils import read_image
from datasets.cityscapes.cityscapes_utils import cityscapes_labels


class CityscapesTestImageDataset(dataset.DatasetMixin):

    """Dataset class for test images of `Cityscapes dataset`_.

    .. _`Cityscapes dataset`: https://www.cityscapes-dataset.com

    .. note::

        Please manually downalod the data because it is not allowed to
        re-distribute Cityscapes dataset.

    Args:
        data_dir (string): Path to the dataset directory. The directory should
            contain at least two directories, :obj:`leftImg8bit` and either
            :obj:`gtFine` or :obj:`gtCoarse`. If :obj:`None` is given, it uses
            :obj:`$CHAINER_DATSET_ROOT/pfnet/chainercv/cityscapes` by default.

    """

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = download.get_dataset_directory(
                'pfnet/chainercv/cityscapes')

        img_dir = os.path.join(data_dir, os.path.join('leftImg8bit', 'test'))
        if not os.path.exists(img_dir):
            raise ValueError(
                'Cityscapes dataset does not exist at the expected location.'
                'Please download it from https://www.cityscapes-dataset.com/.'
                'Then place directory leftImg8bit at {}.'.format(
                    os.path.join(data_dir, 'leftImg8bit')))

        self.img_paths = list()
        for city_dname in sorted(glob.glob(os.path.join(img_dir, '*'))):
            for img_path in sorted(glob.glob(
                    os.path.join(city_dname, '*_leftImg8bit.png'))):
                self.img_paths.append(img_path)

    def __len__(self):
        return len(self.img_paths)

    def get_example(self, i):
        """Returns the i-th test image.

        Returns a color image. The color image is in CHW format.

        Args:
            i (int): The index of the example.

        Returns:
            A color image whose shape is (3, H, W). H and W are height and
            width of the image.
            The dtype of the color image is :obj:`numpy.float32`.

        """
        return read_image(self.img_paths[i])
