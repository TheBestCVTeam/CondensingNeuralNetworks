import logging

from torch import Tensor

from src.filter.base_filter import BaseFilter
from src.utils.log import log


class Diffusion(BaseFilter):
    def short_name(self) -> str:
        return 'u'

    def execute(self, img: Tensor) -> Tensor:
        """
        Dummy does nothing. Only meant to trigger use of precomputed images
        as code is written in matlab
        """
        log('Dummy code executed. No filter applied', logging.ERROR)
        return img
