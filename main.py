import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.sle_mlp import SLEMLP

model_factory = {
    'slemlp': SLEMLP,
}
epochs = {'slemlp': 100}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=list(model_factory.keys()), type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    model = model_factory[args.model]()
    max_epochs = epochs[args.model]
    trainer = pl.Trainer(max_epochs=max_epochs, gpus=1, default_root_dir='./runs/%s' % args.model)
    trainer.fit(model)


if __name__ == '__main__':
    main()