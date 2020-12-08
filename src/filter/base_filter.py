from abc import abstractmethod

from torch import Tensor


class BaseFilter:
    @abstractmethod
    def execute(self, img: Tensor) -> Tensor:
        pass

    @abstractmethod
    def short_name(self) -> str:
        pass

    def long_name(self) -> str:
        return str(self.__class__)
