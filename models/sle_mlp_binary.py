"""
Multilayer Perceptron Model
binary classification of active vs inactive SLE
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import io
from PIL import Image
from dataloader import *
from pytorch_lightning.metrics.classification import AUROC
from models.sle_mlp import init_weights, SLEMLP


def plot_conf_matrix(model, split="val"):
    """
    Add confusion matrix plot to tensorboard

    :param model: the model object
    :param split: string, "train" or "val"
    :return: None
    """
    # from https://stackoverflow.com/questions/65498782/how-to-dump-confusion-matrix-using-tensorboard-logger-in-pytorch-lightning
    tb = model.logger.experiment
    cm = model.train_confusion if (split == 'train') else model.val_confusion
    cm = cm.compute().detach().cpu().numpy().astype(np.int)
    df_cm = pd.DataFrame(cm, index=['inactiveSLE', 'activeSLE'],
                         columns=['inactiveSLE', 'activeSLE'])

    # plot confusion matrix
    plt.figure()
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    tb.add_image(f"{split}_confusion_matrix", im, global_step=model.current_epoch)

    # clear confusion matrix
    if split == 'train':
        model.train_confusion = pl.metrics.classification.ConfusionMatrix(num_classes=2)
    else:
        model.val_confusion = pl.metrics.classification.ConfusionMatrix(num_classes=2)


class SLEMLP_BINARY(SLEMLP):
    """
    MLP model for predicting SLE disease activity
    """

    def __init__(self):
        super().__init__()
        self.out_layer = nn.Conv1d(in_channels=128, out_channels=2, kernel_size=1)
        init_weights(self.out_layer)

        # initialize metrics
        self.acc = pl.metrics.Accuracy()
        self.auroc = AUROC(num_classes=2, pos_label=1)
        self.val_confusion = pl.metrics.classification.ConfusionMatrix(num_classes=2)
        self.train_confusion = pl.metrics.classification.ConfusionMatrix(num_classes=2)

    def training_step(self, batch, batch_idx):
        """
        A single training step on the given batch

        :param batch: current sample batch
        :param batch_idx: current batch id
        :return: a dictionary of training logs, e.g. loss, accuracy
        """
        x, label = batch
        logits = self(x)  # forward step

        # calculate loss and use max logit as predicted value
        weights = torch.tensor([0.3, 0.7], dtype=torch.float)  # increase loss weight of the active SLE class
        loss = F.cross_entropy(logits, label, weights)
        prob = F.softmax(logits, dim=1)
        _, y_hat = torch.max(logits, dim=1)

        # calculate metrics
        train_acc = self.acc(y_hat, label)
        train_auc = self.auroc(prob, label)
        self.train_confusion.update(y_hat, label)

        self.log('loss', loss)
        return {'loss': loss, 'train_auc': train_auc, 'train_acc': train_acc}

    def training_epoch_end(self, outputs):
        """
        Calculate average loss, accuracy and AUC after one epoch of training

        :param outputs: training logs for each batch of the current epoch
        :return: logging results for the entire epoch
        """
        # calculate average loss, accuracy and AUC on the training set
        avg_train_loss = torch.stack([x['loss'].detach() for x in outputs]).mean()
        avg_train_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
        avg_train_auc = torch.stack([x['train_auc'] for x in outputs]).mean()

        # plot confusion matrix
        plot_conf_matrix(self, split='train')

        # manual log since training_epoch_end does not allow return value
        self.log('avg_train_loss', avg_train_loss)
        self.log('avg_train_acc', avg_train_acc)
        self.log('avg_train_auc', avg_train_auc)
        return

    def validation_epoch_end(self, outputs):
        """
        Calculate average loss, accuracy and AUC after one epoch of validation

        :param outputs: validation logs for each batch of the current epoch
        :return: logging results for the entire epoch
        """
        # calculate average loss, accuracy and AUC on the dev set
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_auc = torch.stack([x['val_auc'] for x in outputs]).mean()

        # calculate confusion matrix
        plot_conf_matrix(self)

        logs = {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc, 'avg_val_auc': avg_auc}
        return {'log': logs, 'val_loss': avg_loss, 'progress_bar': logs}

    def train_dataloader(self):
        """
        Load training data

        :return: training data object
        """
        return build_dataloader(split='train', binary=True)

    def val_dataloader(self):
        """
        Load validation data

        :return: validation data object
        """
        return build_dataloader(split='dev', binary=True)


class SLEMLP_KIDNEY_BINARY(SLEMLP_BINARY):
    """
    MLP model for predicting SLE kidney symptom
    """

    def train_dataloader(self):
        """
        Load training data

        :return: training data object
        """
        return build_dataloader(split='train', binary=True, kidney=True)

    def val_dataloader(self):
        """
        Load validation data

        :return: validation data object
        """
        return build_dataloader(split='dev', binary=True, kidney=True)
