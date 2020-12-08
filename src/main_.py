import logging
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim
import yaml
from torch.utils.data import DataLoader

from src.dataset.the_dataset import TheDataset
from src.eval.evaluator import Evaluator
from src.filter.base_filter import BaseFilter
from src.local.loc_folders import LocFolders
from src.models.AENet import AENet
from src.models.trainNet import (get_torch_device, pretrain,
                                 set_random_weights,
                                 train_model)
from src.utils.config import Conf
from src.utils.enums import TDataUse
from src.utils.log import check_error, log, setup_log
from src.utils.misc_func import (get_log_folder, get_run_timestamp,
                                 mkdir_w_par, public_members_as_dict,
                                 strs_to_classes)
from src.utils.stopwatch import StopWatch
from src.utils.train_param import TrainParam

_stopwatch_main: StopWatch
_papers_checkpoint: dict  # Updated in function call
run_timestamp = None


def initialize():
    global run_timestamp
    run_timestamp = get_run_timestamp()
    folder = get_log_folder()
    filename = f'{folder}{run_timestamp} run.log'
    setup_log(filename)  # Only needs to be run once (not needed in each file)
    new_stopwatch()


def new_stopwatch():
    global _stopwatch_main
    _stopwatch_main = StopWatch('Main')


def finalize():
    """
    Closes up things from the run like
        - Stop the main timer
        - Marks the log end
        - Check if there were errors and inform user
    :return:
    """
    _stopwatch_main.end()
    log('\n<<<<<<<<<<<<<<<<<<<<<< COMPLETED >>>>>>>>>>>>>>>>>>>>>>')
    check_error()


def get_output_folder():
    root_folder = Conf.Misc.OUTPUT_FOLDER
    return f'{root_folder}{run_timestamp}/'


def get_dataset(data_use: TDataUse, filters: List[BaseFilter.__class__],
                *, max_size: int = None, should_save_img_after_filters=False):
    base_folder = LocFolders.BASE_WORKING_DATASET_FOLDER
    conf = Conf.CreateBundle.FileLocations

    if data_use == TDataUse.TRAIN:
        fn = base_folder + conf.OUT_TRAIN_LIST_FILENAME
    else:
        fn = base_folder + conf.OUT_TEST_LIST_FILENAME

    return TheDataset(fn, filters, max_size=max_size,
                      should_save_img_after_filters=should_save_img_after_filters)


def get_dataloader(filters: List[BaseFilter.__class__] = None,
                   should_save_img_after_filters=False):
    if filters is None:
        filter_list = Conf.RunParams.MODEL_TRAIN_DEFAULTS['filters']
        filters = strs_to_classes(filter_list)
    conf = Conf.RunParams.DataLoader

    # Get Datasets - Capping the size of the validate set to one batch so
    # that validation is done on the same set of data on every iteration
    train_ds = get_dataset(TDataUse.TRAIN, filters,
                           should_save_img_after_filters=True)
    val_ds = get_dataset(TDataUse.VALIDATION, filters,
                         max_size=conf.BatchSizes.VALIDATION)
    test_ds = get_dataset(TDataUse.TEST, filters)

    # Ensure train set is pre-computed otherwise pre-compute it
    train_ds.precompute()

    # Make data loaders from datasets
    loader_train = DataLoader(dataset=train_ds,
                              batch_size=conf.BatchSizes.TRAIN,
                              shuffle=True,
                              num_workers=conf.NUM_WORKERS)
    loader_val = DataLoader(dataset=val_ds,
                            batch_size=conf.BatchSizes.VALIDATION,
                            shuffle=True,
                            num_workers=conf.NUM_WORKERS)
    loader_test = DataLoader(dataset=test_ds, batch_size=conf.BatchSizes.TEST,
                             shuffle=False, num_workers=conf.NUM_WORKERS)

    return loader_train, loader_val, loader_test


def load_checkpoint_from_paper():
    global _papers_checkpoint
    _papers_checkpoint = \
        torch.load('./src/pickle/ckpt_iter.pth.tar',
                   map_location=get_torch_device())


def generate_trained_model(train_param: TrainParam) -> Tuple[
    nn.Module, DataLoader, StopWatch, pd.DataFrame]:
    loader_train, loader_val, loader_test = get_dataloader(train_param.filters)
    model = AENet(num_classes=2,
                  num_cnn_layers=train_param.cnn_count,
                  relevant_layers=True,
                  dl_bypass_input=train_param.dl_bypass_input)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=train_param.learn_rate)

    model = model.to(get_torch_device())
    if train_param.should_pretrain:
        pretrain(model, _papers_checkpoint['state_dict'])
        # model.set_train_cnn_layers(train=False)
    else:
        set_random_weights(model)

    # Expectation is that the training dataset has be precomputed (Track
    # starting value to determine if the assumption was wrong)
    start_filter_compute = loader_train.dataset.img_count_filters_applied

    sw_train, training_stats = train_model(model, optimizer,
                                           loader_train=loader_train,
                                           loader_val=loader_val,
                                           epochs=train_param.epochs,
                                           model_caption=str(train_param),
                                           max_epochs_before_progress=
                                           train_param.
                                           max_epochs_before_progress,
                                           max_total_epochs_no_progress=
                                           train_param.
                                           max_total_epochs_no_progress)
    if loader_train.dataset.img_count_filters_applied != start_filter_compute:
        log(
            f'Expected dataset to have been precomputed but at start there '
            f'were {start_filter_compute} and the end there were '
            f'{loader_train.dataset.img_count_filters_applied}', logging.ERROR)

    return model, loader_test, sw_train, training_stats


def yaml_to_file(data, filename):
    folder = get_output_folder()
    mkdir_w_par(folder)
    fn = folder + filename
    yaml_data = yaml.dump(data)
    with open(fn, 'w') as f:
        f.write(yaml_data)


def csv_to_file(df, filename):
    folder = get_output_folder()
    mkdir_w_par(folder)
    fn = folder + filename
    df.to_csv(fn, index=False)


def train_and_evaluate_models():
    evaluations = []
    output_folder = get_output_folder()
    separator = '-' * 80
    separator = f'\n{separator}\n'
    export_df = pd.DataFrame(
        columns=["test_name", "roc_auc", "accuracy", "fpr",
                 "tpr", "train_t", "eval_t", "total_t"])
    for value in Conf.RunParams.MODEL_TRAIN:
        params = TrainParam(**value)
        sw_test = StopWatch(f'{params}')
        model, loader_test, sw_train, training_stats = generate_trained_model(
            params)
        evaluation = Evaluator(model, loader_test, params, output_folder,
                               training_stats)
        log(evaluation)
        evaluations.append(evaluation)
        sw_test.end()
        log(separator)

        # Save Evaluations to disk
        yaml_to_file(evaluations, Conf.Misc.EVALUATIONS_FILENAME)

        # Convert to CSV
        export_df = export_df.append({"test_name": str(params),
                                      "roc_auc": evaluation.roc_auc,
                                      "accuracy": evaluation.accuracy,
                                      "fpr" : evaluation.fpr,
                                      "tpr" : evaluation.tpr,
                                      "train_t" : sw_train.as_float(),
                                      "eval_t" : evaluation.sw_scoring.as_float(),
                                      "eval_network_t": evaluation.total_time_network_eval,
                                      "total_t" : sw_test.as_float()
                                      }, ignore_index=True)
        csv_to_file(export_df, Conf.Misc.EVALUATIONS_CSV_FILENAME)


def finalize_and_save_config():
    # Insert IDs
    for i, value in enumerate(Conf.RunParams.MODEL_TRAIN):
        value["id_"] = i

    settings = public_members_as_dict(Conf.RunParams)
    yaml_to_file(settings, Conf.Misc.SETTINGS_FILENAME)


def main():
    initialize()

    finalize_and_save_config()

    load_checkpoint_from_paper()

    train_and_evaluate_models()

    finalize()


if __name__ == '__main__':
    main()
