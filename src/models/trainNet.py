import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch._six import inf
from torch.nn.functional import binary_cross_entropy_with_logits

from src.utils.config import Conf
from src.utils.log import log
from src.utils.stopwatch import StopWatch


def get_torch_device():
    if Conf.Training.ALLOW_GPU and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def pretrain(model, state_dict, fc_set=True):
    own_state = model.state_dict()

    for name, param in state_dict.items():
        real_name = name.replace('module.', '')

        # break after adding cnn layer weights
        if "fc_live" in real_name:
            break

        if real_name in own_state:
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[real_name].copy_(param)
            except Exception as e:
                log(f'While copying the parameter named {real_name}, '
                    f'whose dimensions in the model are '
                    f'{own_state[name].size()} and '
                    f'whose dimensions in the checkpoint are {param.size()}.\n'
                    f'ErrMsg: {e}'
                    , logging.ERROR)
                log("But don't worry about it. Continue pretraining.")

    # set weights of fc_live
    if fc_set:
        num_feats = model.fc_live.in_features
        fc2_w = random_weight((num_feats, 2))
        fc2_b = zero_weight((2,))

        model.fc_live = nn.Linear(num_feats, 2)
        model.fc_live.weights = nn.Parameter(fc2_w)
        model.fc_live.bias = nn.Parameter(fc2_b)


def set_random_weights(model):
    for layer_name in model.state_dict():
        layer = model.state_dict()[layer_name]
        if '.weight' in layer_name:
            rand_weights = random_weight(layer.shape)
            # rand_weights = zero_weight(layer.shape)
            layer.copy_(nn.Parameter(rand_weights))
        elif ".bias" in layer_name:
            zero_bias = zero_weight(layer.shape)
            layer.copy_(nn.Parameter(zero_bias))


def random_weight(shape):
    """
    Create random Tensors for weights; setting requires_grad=True means that we
    want to compute gradients for these Tensors during the backward pass.
    We use Kaiming normalization: sqrt(2 / fan_in)
    """
    if len(shape) == 2:  # FC weight
        fan_in = shape[0]
    else:
        fan_in = np.prod(
            shape[1:])  # conv weight [out_channel, in_channel, kH, kW]
    # randn is standard normal distribution generator.
    w = torch.randn(shape, device=get_torch_device(),
                    dtype=Conf.Training.DTYPE) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w


def zero_weight(shape):
    return torch.zeros(shape, device=get_torch_device(),
                       dtype=Conf.Training.DTYPE, requires_grad=True)


# noinspection PyAbstractClass
class Flatten(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return flatten(x)


def flatten(x):
    n = x.shape[0]  # read in N, C, H, W
    return x.view(n, -1)  # "flatten" the C * H * W


def check_accuracy(loader, model):
    """
    Checks accuracy on all the data from the data loader

    :param loader: the loader (Expected to provide 1 batch of the full data)
    :param model: The model to be tested
    :return: Accuracy, Scores, y_test (The ground truth labels)
    """
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    scores_all = None
    y_test = None
    total_time_network = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=get_torch_device(), dtype=Conf.Training.DTYPE)
            y = y.to(device=get_torch_device(), dtype=torch.long)

            # Accumulate scores
            start_time = time.time()
            scores_x = model(x)
            total_time_network += (
                time.time() - start_time)  # add time it takes to run
            # evaluation

            if scores_all is None:
                scores_all = scores_x
            else:
                scores_all = torch.cat([scores_all, scores_x], dim=0)

            # Accumulate labels
            if y_test is None:
                y_test = y
            else:
                y_test = torch.cat([y_test, y], dim=0)
            _, preds = scores_x.max(1)
            # noinspection PyUnresolvedReferences
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        acc = float(num_correct) / num_samples
        log('Got %d / %d correct (%.2f)' % (
            num_correct, num_samples, 100 * acc))
        return acc, scores_all, y_test, total_time_network


def train_model(model, optimizer, loader_train, loader_val, epochs,
                model_caption, max_epochs_before_progress,
                max_total_epochs_no_progress):
    """
    Trains model and output accuracies as it goes
    :param max_total_epochs_no_progress: Total number of epochs where model is
    allowed to not make progress before training is terminated
    :param max_epochs_before_progress: number of epochs before model must make
    progress for training to continue
    :param model: The model to be trained
    :param optimizer: The optimizer to use
    :param loader_train: Provides the data to train on
    :param loader_val: Provides the validation date for accuracy output
    :param epochs: Number of epochs to train for
    :param model_caption: Display name for the model
    :return:
    """
    conf = Conf.RunParams
    log(
        f'Train {model_caption} for {epochs} epochs.')
    sw = StopWatch(
        f"Train Model - {model_caption}")

    device = get_torch_device()
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    # Setup variables to track performance for early training stop
    consecutive_epochs_no_progress = 0
    total_epochs_no_progress = 0
    last_eval_score = -inf

    training_stats = pd.DataFrame(columns=["epoch", "loss", "acc"])
    for e in range(epochs):
        sw_epoch = StopWatch(f'Epoch: {e}', start_log_level=logging.DEBUG)
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode

            x = x.to(device=device, dtype=Conf.Training.DTYPE)
            y = y.to(device=device, dtype=torch.long)

            y_log = torch.zeros(y.shape[0], 2, device=device)
            y_log[range(y_log.shape[0]), y] = 1
            scores = flatten(model(x))

            loss = binary_cross_entropy_with_logits(scores, y_log)

            # Zero out all of the gradients for the variables which the
            # optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

        log(f'Epoch: {e}, loss = %.4f' % (loss.item()))
        acc, _, _, _ = check_accuracy(loader_val, model)
        training_stats = training_stats.append({"epoch": e,
                                                "loss": loss.item(),
                                                "acc": acc},
                                               ignore_index=True)
        if (acc - last_eval_score) < conf.TRAIN_MIN_PROGRESS:
            consecutive_epochs_no_progress += 1
            total_epochs_no_progress += 1
        else:
            consecutive_epochs_no_progress = 0
        last_eval_score = acc
        sw_epoch.end()
        if consecutive_epochs_no_progress >= max_epochs_before_progress:
            log(
                f'Stopping training at {e} epochs because of insufficient '
                f'progress')
            break
        if total_epochs_no_progress >= max_total_epochs_no_progress:
            log(
                f'Stopping training at {e} epochs because failed to make '
                f'progress too many times')
            break
    sw.end()
    return sw, training_stats
