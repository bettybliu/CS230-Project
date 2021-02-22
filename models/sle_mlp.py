"""
Multilayer Perceptron Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from dataloader import *
from pytorch_lightning.metrics.classification import AUROC


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
        nn.ReLU(True)
    ]
    model = nn.Sequential(*layers)
    init_weights(model)
    return model


def init_weights(model):
    """
    Initialize weights using xavier initialization.
    Initialize all biases to zero.

    :param model: the model block to be initialized
    :return: None
    """
    for m in model.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)


class SLEMLP(pl.LightningModule):
    """
    MLP model for predicting SLE disease activity
    """
    def __init__(self):
        super().__init__()
        self.proj_layer = build_mlp_block(in_c=8, out_c=1)
        self.mlp1 = build_mlp_block(in_c=18190, out_c=512)
        self.mlp2 = build_mlp_block(in_c=512, out_c=128)
        self.out_layer = nn.Conv1d(in_channels=128, out_channels=3, kernel_size=1)
        init_weights(self.out_layer)

        self.acc = pl.metrics.Accuracy()
        self.auroc = AUROC(num_classes=3, pos_label=1)

    def forward(self, x):
        """
        Forward propagation step

        :param x: input data of shape: batchsize x 8 x 18190
        :return: logits: batchsize x 3
        """
        x = self.proj_layer(x)
        x = x.transpose(1, 2).contiguous()  # batchsize x 18190 x 1
        x = self.mlp2(self.mlp1(x))
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
        logits = self(x) # forward step

        # calculate loss and use max logit as predicted value
        loss = F.cross_entropy(logits, label)
        prob = F.softmax(logits, dim=1)
        _, y_hat = torch.max(logits, dim=1)

        # calculate metrics
        train_acc = self.acc(y_hat, label)
        train_auc = self.auroc(prob, label)

        tensorboard_logs = {'train_loss': loss, 'train_acc': train_acc, 'train_auc': train_auc}
        return {'log': tensorboard_logs, 'train_loss': loss, 'train_auc': train_auc}

    def validation_step(self, batch, batch_idx):
        """
        A single training step on the given batch

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

        logs = {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc, 'avg_val_auc': avg_auc}
        return {'log': logs, 'val_loss': avg_loss, 'progress_bar': logs}

    def configure_optimizers(self):
        """
        Setup Adam optimizer and learning rate decay

        :return: optimizer function and the learning rate scheduler function
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_func = lambda epoch: 2**(-1 * (epoch // 20))
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

    def transfer_batch_to_device(self, batch, device=None):
        """
        Interface with GPU/CPU

        :param batch: current batch data
        :param device: CPU or GPU device
        :return:
        """
        points, target = batch
        target = target.long()

        batch = (points, target)
        return super().transfer_batch_to_device(batch, device)