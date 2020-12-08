import logging

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader

from src.models.trainNet import check_accuracy
from src.utils.config import Conf
from src.utils.misc_func import mkdir_w_par
from src.utils.stopwatch import StopWatch
from src.utils.train_param import TrainParam


class Evaluator:
    def __init__(self, model: nn.Module, loader: DataLoader,
                 params: TrainParam, output_folder,
                 training_stats: pd.DataFrame):
        """
        Evaluates the model passed using the loader and stores the results
        in its fields. It treats the spoof class as the positive class for
        true positive rate calculations
        :param model: The model to be evaluated
        :param loader: Point to the data to be used for evaluation
        :param params: The training parameters for the model to identify it
        """
        self.model_caption = str(params)
        sw = StopWatch(f'Evaluation - {self.model_caption}')

        # Set and create output folder
        self.output_folder = f'{output_folder}{self.model_caption}/'
        mkdir_w_par(self.output_folder)

        # Get Accuracy,  Scores and Ground True Labels
        self.sw_scoring = StopWatch(f'Scoring', start_log_level=logging.DEBUG)
        accuracy, scores, y_test, total_time_network_eval = check_accuracy(
            loader, model)
        self.sw_scoring.end()
        self.accuracy = accuracy
        self.total_time_network_eval = total_time_network_eval

        # Convert scores into 1-d array by taking diff between classes
        scores = scores[:, 1] - scores[:, 0]

        # load tensor into CPU for numpy conversion
        y_test = y_test.to(device=torch.device('cpu'))
        scores = scores.to(device=torch.device('cpu'))

        # Get False Positive, True Positive for the possible thresholds
        fpr_roc, tpr_roc, thresholds = metrics.roc_curve(y_test, scores)

        # Save number of different thresholds found
        self.threshold_count = len(thresholds)

        # Calculate and store FPR and TRP
        fpr, tpr = self.get_fpr_tpr(fpr_roc, tpr_roc, thresholds)
        self.fpr = float(fpr)
        self.tpr = float(tpr)

        # Calculate AUC
        self.roc_auc = float(metrics.auc(fpr_roc, tpr_roc))

        # Plot ROC
        self.plot_roc(fpr_roc, tpr_roc)

        # Plot Training Info and write to file
        self.plot_training_info(training_stats)
        self.training_info = training_stats.to_dict()

        # Save a copy of the model structure as text
        self.save_model_structure(model)

        # Save actual model
        if Conf.RunParams.SAVE_EACH_FINAL_MODEL:
            torch.save(model,
                       self.output_folder + Conf.Eval.MODEL_FILENAME)

        sw.end()

    def plot_roc(self, fpr, tpr):
        plt.figure(dpi=300)
        plt.plot(fpr, tpr, color='darkorange',
                 label='ROC curve (area = %0.2f)' % self.roc_auc)
        plt.plot([0, 1], [0, 1], label='Chance', color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.model_caption} Receiver operating characteristic')
        plt.legend(loc="lower right")

        # Save plot to file
        fn = self.output_folder + Conf.Eval.ROC_FILENAME
        plt.savefig(fn)

        plt.close()

    def plot_training_info(self, training_info: pd.DataFrame):
        plt.figure(dpi=300)
        plt.plot(training_info["loss"], label='Loss', color='red')
        plt.plot(training_info["acc"], label='Accuracy', color='green',
                 linestyle='--')
        plt.ylim([0.0, 1.10])
        plt.legend()

        # Save plot to file
        fn = self.output_folder + Conf.Eval.TRAINING_INFO_FILENAME
        plt.savefig(fn)

        plt.close()

    def __str__(self):
        return yaml.dump(self)

    @staticmethod
    def get_fpr_tpr(fpr_roc, tpr_roc, thresholds):
        """
        Calculate FPR and TPR based on score > 0 being positive. Assumes
        y_test is only 1 or 0. Selects the last threshold before going
        negative if one exists.

        :param fpr_roc: list of fpr rates corresponding to the thresholds
        :param tpr_roc: list of tpr rates corresponding to the thresholds
        :param thresholds: list of thresholds from ROC. Expects values to be
        in monotonically decreasing order (Should be strictly decreasing but
        not required)
        :return: FPR and TPR
        """

        if len(thresholds) <= 0 or thresholds[0] <= 0:
            fpr = 0
            tpr = 0
        else:
            last_ind = -1
            for i, curr in enumerate(thresholds):
                if curr > 0:
                    last_ind = i
                else:
                    break
            if last_ind < 0:
                fpr = 0
                tpr = 0
            else:
                fpr = fpr_roc[last_ind]
                tpr = tpr_roc[last_ind]

        return fpr, tpr

    def save_model_structure(self, model):
        fn = self.output_folder + Conf.Eval.MODEL_AS_STR_FILENAME
        with open(fn, 'w') as f:
            f.write(str(model))
