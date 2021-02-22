import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from dataloader import *


def build_mlp_block(in_c, out_c, kernel_size=1):
    layers = [
        nn.Conv1d(in_c, out_c, kernel_size=kernel_size),
        nn.BatchNorm1d(out_c),
        nn.ReLU(True)
    ]
    model = nn.Sequential(*layers)
    init_weights(model)
    return model


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)


class SLEMLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.proj_layer = build_mlp_block(in_c=8, out_c=1)
        self.mlp1 = build_mlp_block(in_c=18190, out_c=512)
        self.mlp2 = build_mlp_block(in_c=512, out_c=128)
        self.out_layer = nn.Conv1d(in_channels=128, out_channels=3, kernel_size=1)
        init_weights(self.out_layer)

        self.acc = pl.metrics.Accuracy()

    def forward(self, x):
        """
        :param x: input data of shape: bs x 8 x 18190
        :return: logits: bs x 3
        """
        x = self.proj_layer(x)
        x = x.transpose(1, 2).contiguous()  # bs x 18190 x 1
        x = self.mlp2(self.mlp1(x))
        logits = self.out_layer(x).squeeze(dim=2)
        return logits

    def training_step(self, batch, batch_idx):
        x, label = batch
        logits = self(x)

        loss = F.cross_entropy(logits, label)

        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, label = batch
        logits = self(x)

        loss = F.cross_entropy(logits, label)
        _, y_hat = torch.max(logits, dim=1)
        test_acc = self.acc(y_hat, label)
        return {'val_loss': loss, 'val_acc': test_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        logs = {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc}
        return {'val_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_func = lambda epoch: 2**(-1 * (epoch // 20))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return build_dataloader(split='train')

    def val_dataloader(self):
        return build_dataloader(split='test')

    def transfer_batch_to_device(self, batch, device=None):
        points, target = batch
        target = target[:, 0].long()

        batch = (points, target)
        return super().transfer_batch_to_device(batch, device)