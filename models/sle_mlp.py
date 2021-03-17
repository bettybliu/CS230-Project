"""
Multilayer Perceptron Model
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
from pytorch_lightning.metrics.classification import AUROC, ConfusionMatrix, ROC


def build_mlp_block(in_c, out_c, kernel_size=1):
    """
    Build a single MLP block with three layers:
        convolution, batchnorm and relu activation

    :param in_c: number of input channels
    :param out_c: number of output channels
    :param kernel_size: convolution filter size f
    :return: the MLP block model
    """
    layers = [
        nn.Conv1d(in_c, out_c, kernel_size=kernel_size),
        nn.BatchNorm1d(out_c),
        nn.LeakyReLU(0.1)
    ]
    model = nn.Sequential(*layers)
    init_weights(model)
    return model


def init_weights(model):
    """
    Initialize weights using He initialization.
    Initialize all biases to zero.

    :param model: the model block to be initialized
    :return: None
    """
    for m in model.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)


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
    df_cm = pd.DataFrame(cm, index=['inactiveSLE', 'activeSLE', 'healthy'],
                         columns=['inactiveSLE', 'activeSLE', 'healthy'])

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
        model.train_confusion = ConfusionMatrix(num_classes=3)
    else:
        model.val_confusion = ConfusionMatrix(num_classes=3)

def plot_roc(model, split="val", auc=0):
    """
    Add ROC curve to tensorboard

    :param model: the model object
    :param split: string, "train" or "val"
    :param auc: calculated AUROC for target class
    :return: None
    """
    tb = model.logger.experiment
    roc = model.train_roc if (split == 'train') else model.val_roc
    fpr, tpr, threshold = [[t.detach().cpu().numpy().astype(np.float) for t in ls] for ls in roc.compute()]

    # plot ROC curve
    fig, ax1 = plt.subplots()
    cls = 1 # plot roc for class 1
    ax1.plot(fpr[cls], tpr[cls], label=f'AUC = {auc:.2f}',
             lw=2)

    # plot random chance line
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
            xlabel=f'False Positive Rate (Positive Label {cls})',
            ylabel=f'True Positive Rate (Positive Label {cls})',
            title="Receiver operating characteristic")
    ax1.legend(loc='lower right')

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    tb.add_image(f"{split}_ROC", im, global_step=model.current_epoch)

    # clear roc
    if split == 'train':
        model.train_roc = ROC(num_classes=3)
    else:
        model.val_roc = ROC(num_classes=3)

class SLEMLP(pl.LightningModule):
    """
    MLP model for predicting SLE disease activity
    """

    def __init__(self, hparams):
        """
        Initialize model attributes

        :param hparams: hyperparameters dictionary
        """
        super().__init__()
        self.hparams = hparams
        self.proj_layer = build_mlp_block(in_c=8, out_c=1)
        self.mlp1 = build_mlp_block(in_c=18190, out_c=512)
        self.mlp2 = build_mlp_block(in_c=512, out_c=128)
        self.out_layer = nn.Conv1d(in_channels=128, out_channels=3, kernel_size=1)
        self.dropout_layer = nn.Dropout(p=hparams['dropout_prob'])
        init_weights(self.out_layer)

        # initialize metrics
        self.acc = pl.metrics.Accuracy()
        self.auroc = AUROC(num_classes=3, pos_label=1)
        self.val_confusion = ConfusionMatrix(num_classes=3)
        self.train_confusion = ConfusionMatrix(num_classes=3)
        self.train_roc = ROC(num_classes=3)
        self.val_roc = ROC(num_classes=3)

    def forward(self, x):
        """
        Forward propagation step

        :param x: input data of shape: batchsize x 8 x 18190
        :return: logits: batchsize x 3
        """
        x = self.proj_layer(x)
        x = x.transpose(1, 2).contiguous()  # batchsize x 18190 x 1
        x = self.mlp2(self.dropout_layer(self.mlp1(x)))
        logits = self.out_layer(x).squeeze(dim=2)
        return logits

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
        #   increase loss weight of the active SLE class
        weights = torch.tensor(self.hparams['loss_cls_weights'], dtype=torch.float)
        loss = F.cross_entropy(logits, label, weights)
        prob = F.softmax(logits, dim=1)
        _, y_hat = torch.max(logits, dim=1)

        # calculate metrics
        train_acc = self.acc(y_hat, label)
        train_auc = self.auroc(prob, label)
        self.train_confusion.update(y_hat, label)
        self.train_roc.update(prob, label)

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

        # plot roc curve
        plot_roc(self, split='train', auc=avg_train_auc)

        # manual log since training_epoch_end does not allow return value
        self.log('avg_train_loss', avg_train_loss)
        self.log('avg_train_acc', avg_train_acc)
        self.log('avg_train_auc', avg_train_auc)
        return

    def validation_step(self, batch, batch_idx):
        """
        A single validation step on the given batch

        :param batch: current sample batch
        :param batch_idx: current batch id
        :return: a dictionary of validation logs, e.g. loss, accuracy
        """
        x, label = batch
        logits = self(x)

        # calculate loss and use max logit as predicted value
        loss = F.cross_entropy(logits, label)
        prob = F.softmax(logits, dim=1)
        _, y_hat = torch.max(logits, dim=1)

        # calculate metrics
        test_acc = self.acc(y_hat, label)
        test_auc = self.auroc(prob, label)
        self.val_confusion.update(y_hat, label)
        self.val_roc.update(prob, label)

        return {'val_loss': loss, 'val_acc': test_acc, 'val_auc': test_auc}

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

        # plot roc curve
        plot_roc(self, auc=avg_auc)

        logs = {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc, 'avg_val_auc': avg_auc}
        return {'log': logs, 'val_loss': avg_loss, 'progress_bar': logs}

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, *args, **kwargs):
        return self.validation_epoch_end(*args, **kwargs)

    def configure_optimizers(self):
        """
        Setup Adam optimizer and learning rate decay

        :return: optimizer function and the learning rate scheduler function
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'],
                                     weight_decay=self.hparams['l2_strength'])
        lr_func = lambda epoch: 2 ** (-1 * (epoch // self.hparams['lr_half_time']))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        """
        Load training data

        :return: training data object
        """
        return build_dataloader(split='train')

    def val_dataloader(self):
        """
        Load validation data

        :return: validation data object
        """
        return build_dataloader(split='dev')

    def test_dataloader(self):
        """
        Load test data

        :return: test data object
        """
        return build_dataloader(split='test')

    def transfer_batch_to_device(self, batch, device=None):
        """
        Interface with GPU/CPU

        :param batch: current batch data
        :param device: CPU or GPU device
        :return: reference to batch data on device
        """
        points, target = batch
        target = target.long()

        batch = (points, target)
        return super().transfer_batch_to_device(batch, device)
