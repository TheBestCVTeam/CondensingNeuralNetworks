from typing import List

from src.utils.config import Conf
from src.utils.misc_func import strs_to_classes


class TrainParam(object):
    def __init__(self, id_, *, cnn_count: int = None,
                 should_pretrain: bool = None,
                 filters: List[str] = None,
                 epochs: int = None,
                 eval_its: int = None,
                 dl_bypass_input: bool = None,
                 learn_rate: float = None,
                 max_epochs_before_progress: int = None,
                 max_total_epochs_no_progress: int = None):
        self.id_ = id_
        conf = Conf.RunParams.MODEL_TRAIN_DEFAULTS
        if cnn_count is None:
            cnn_count = conf['cnn_count']
        if should_pretrain is None:
            should_pretrain = conf['should_pretrain']
        if filters is None:
            filters = conf['filters']
        if epochs is None:
            epochs = conf['epochs']
        if dl_bypass_input is None:
            dl_bypass_input = conf['dl_bypass_input']
        if learn_rate is None:
            learn_rate = conf['learn_rate']
        if max_epochs_before_progress is None:
            max_epochs_before_progress = conf[
                'max_epochs_before_progress']
        if max_total_epochs_no_progress is None:
            max_total_epochs_no_progress = conf['max_total_epochs_no_progress']
        self.cnn_count = cnn_count
        self.should_pretrain = should_pretrain
        self.filters = strs_to_classes(filters)
        self.epochs = epochs
        self.eval_its = eval_its
        self.dl_bypass_input = dl_bypass_input
        self.learn_rate = learn_rate
        self.max_epochs_before_progress = max_epochs_before_progress
        self.max_total_epochs_no_progress = max_total_epochs_no_progress

    def __str__(self):
        return f'id_{self.id_}-cnn_{self.cnn_count}-FT_' \
               f'{"y" if self.dl_bypass_input else "n"}-fil_' \
               f'{self.fil_to_str()}'

    def fil_to_str(self):
        return fil_to_str(self.filters)


def fil_to_str(filters):
    result = ""
    for fil in filters:
        result += fil().short_name()
    return result
