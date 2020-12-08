import math

import torch
import torch.distributions
from torch import Tensor
from torch.nn.functional import conv2d

from src.filter.base_filter import BaseFilter
from src.utils.config import Conf


class DiffGaussian(BaseFilter):
    def short_name(self) -> str:
        return 'd'

    def execute(self, img: Tensor) -> Tensor:
        """
        Create a Difference of Gaussian (DoG) activation layer.
        Filter parameters are defined in the 'config.py' file.
        > sigma1 is the standard deviation of the first Gaussian filter.
        > sigma2 is the standard deviation of the second Gaussian filter.
        :param img: input image in tensor form.
        :return: activation layer of the DoG kernel (luminance image).
        """
        conf = Conf.RunParams.Filters
        filter_size_orig = conf.DiffGaussian.FILTER_SIZE_ORIG
        sigma1 = conf.DiffGaussian.SIGMA1
        sigma2 = conf.DiffGaussian.SIGMA2
        channels = conf.DiffGaussian.CHANNELS
        dim = conf.DiffGaussian.DIM

        def gauss(filter_size_orig_, channels_, dim_, sigma) -> Tensor:
            """
            This creates a 1x2 tensor for the filter_size and sigma which is
            used when creating the filter.
            :param filter_size_orig_: size of the square filter.
            :param channels_: number of channels of the input image (3 for
            RGB).
            :param dim_: spatial dimensions of the input image.
            :param sigma: standard deviation of the Gaussian filter.
            :return:
            """
            filter_size = [filter_size_orig_] * dim_
            sigma = [sigma] * dim_

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

            # This part reshapes the filter to be the appropriate size for
            # convolution.
            gfilter = gfilter.view(1, 1, *gfilter.size())
            gfilter = gfilter.repeat(channels_, *[1] * (gfilter.dim() - 1))

            return gfilter

        # This part creates the two filters required for the DoG,
        # then computes the DoG.
        gfilter1 = gauss(filter_size_orig, channels, dim, sigma1)
        gfilter2 = gauss(filter_size_orig, channels, dim, sigma2)
        dog_filter = torch.sub(gfilter2, gfilter1)

        # This part calculates the necessary padding for the filters in both
        # spatial directions.
        stride = 1
        pad_w = math.ceil(
            ((stride - 1) * img.shape[1] - stride + filter_size_orig) / 2)
        pad_h = math.ceil(
            ((stride - 1) * img.shape[2] - stride + filter_size_orig) / 2)

        # This part convolves the image with the filter.
        # The unsqueeze is because conv2d expects a batch dimension for the
        # images as input.
        # The squeeze is because we no longer require the batch dimension
        # for printing.
        imgx = conv2d(img.unsqueeze_(0),
                      weight=dog_filter,
                      groups=channels,
                      stride=stride,
                      padding=(pad_h, pad_w)).squeeze_(0)

        # Edge detection with additive activation layer for retaining colour.
        # Get luminance values of the edge detection activation layer.
        imgx = sum(imgx, 0) / 3

        imgx = imgx / torch.max(
            imgx)  # normalize by max value in the edge detection layer
        threshold = 10 * torch.mean(
            imgx)  # set threshold value based on mean of the edge detection
        # layer and constant factor
        binary = 1.0 * (
            imgx > threshold)  # convert the edge layer to binary image
        # based on threshold
        binary = 1 - binary  # invert the binary layer so the outline is black
        imout = binary * img  # apply the binary layer to original image (
        # overlay)

        return imout.squeeze_(0)
