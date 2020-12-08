import math

import torch
import torch.distributions
from torch import Tensor
from torch.nn.functional import conv2d

from src.filter.base_filter import BaseFilter
from src.utils.config import Conf


class Gaussian(BaseFilter):
    def short_name(self) -> str:
        return 'g'

    def execute(self, img: Tensor) -> Tensor:
        """
        Create a Gaussian activation layer.
        Filter parameters are defined 'config.py' file.
        > filter_size_orig specifies the size of the square filter.
        > sigma is the standard deviation of the Gaussian filter.
        > channels corresponds to each of R, G and B for a given image.
        > dim is the number of spatial dimensions.
        :param img:
        :return: The modified image in tensor form.
        """
        conf = Conf.RunParams.Filters
        filter_size_orig = conf.Gaussian.FILTER_SIZE_ORIG
        sigma = conf.Gaussian.SIGMA
        channels = conf.Gaussian.CHANNELS
        dim = conf.Gaussian.DIM

        # This creates a 1x2 tensor for the filter_size and sigma which is
        # used when creating the filter.
        filter_size = [filter_size_orig] * dim
        sigma = [sigma] * dim

        # This part creates the gfilter in each spatial direction.
        gfilter = 1
        xmesh = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in filter_size
            ]
        )
        for size, std, x in zip(filter_size, sigma, xmesh):
            mean = (size - 1) / 2
            coeff = 1 / (std * math.sqrt(2 * math.pi))
            expo = torch.exp(-((x - mean) ** 2 / (2 * std ** 2)))
            gfilter *= coeff * expo

        # This part sets the sum of all values in the filter to be 1.
        gfilter = gfilter / torch.sum(gfilter)

        # This part reshapes the filter to be the appropriate size for conv2d.
        gfilter = gfilter.view(1, 1, *gfilter.size())
        gfilter = gfilter.repeat(channels, *[1] * (gfilter.dim() - 1))

        # This part calculates the necessary padding for the filter in both
        # spatial directions.
        stride = 1
        pad_w = math.ceil(
            ((stride - 1) * img.shape[1] - stride + filter_size_orig) / 2)
        pad_h = math.ceil(
            ((stride - 1) * img.shape[2] - stride + filter_size_orig) / 2)

        # This part convolves the image with the filter.
        # The unsqueeze is necessary because conv2d expects a batch
        # dimension for the images as input.
        # The squeeze is necessary because we no longer require the batch
        # dimension, and the image
        # printer expects a 3xWxH image.
        imgx = conv2d(img.unsqueeze_(0),
                      weight=gfilter,
                      groups=channels,
                      stride=stride,
                      padding=(pad_h, pad_w)).squeeze_(0)

        # log("Gaussian: ", imgx.shape)
        return imgx
