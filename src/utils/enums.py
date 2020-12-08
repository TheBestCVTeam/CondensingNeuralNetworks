from enum import Enum, auto


class TDataLabel(Enum):
    LIVE = 0
    SPOOF = 1


class TDataUse(Enum):
    TRAIN = auto()
    TEST = auto()
    VALIDATION = auto()
