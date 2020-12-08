import torch
import torch.distributions
from torch import Tensor

from src.filter.base_filter import BaseFilter


class Histeq(BaseFilter):
    def short_name(self) -> str:
        return 'h'

    def execute(self, img: Tensor) -> Tensor:
        """
        Histogram equalize the input using scaled CDF.
        :param img: input.
        :return: histogram equalized input.
        """
        width = img.shape[1]
        height = img.shape[2]
        num_pixels = width * height

        # Create an index to be used while sampling scaled CDF.
        index_r = (255 * img[0, :, :]).tolist()
        index_g = (255 * img[1, :, :]).tolist()
        index_b = (255 * img[2, :, :]).tolist()

        # Create histograms of the input pixels.
        r_channel_hist = torch.histc(img[0, :, :], bins=256, min=0, max=1)
        g_channel_hist = torch.histc(img[1, :, :], bins=256, min=0, max=1)
        b_channel_hist = torch.histc(img[2, :, :], bins=256, min=0, max=1)

        # Compute the CDF of the image.
        r = 255 * torch.cumsum(r_channel_hist, dim=0) / num_pixels
        g = 255 * torch.cumsum(g_channel_hist, dim=0) / num_pixels
        b = 255 * torch.cumsum(b_channel_hist, dim=0) / num_pixels

        # Sample the scaled CDF for the pixel values in the image.
        out_r = (r[index_r]).type(torch.float32)
        out_g = (g[index_g]).type(torch.float32)
        out_b = (b[index_b]).type(torch.float32)

        # This cast needs to be made in order to fix pipeline issues
        # (error thrown when passing this to the Gaussian or DiffGaussian
        # filters).
        imgx = torch.stack((out_r, out_g, out_b), 0) / 255
        # log("HistEQ: ", imgx.shape)

        return imgx
