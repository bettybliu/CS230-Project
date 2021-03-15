import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.sle_mlp import SLEMLP
from models.sle_mlp_binary import SLEMLP_BINARY, SLEMLP_KIDNEY_BINARY

model_factory = {
    'slemlp': SLEMLP,
    'slemlp_binary': SLEMLP_BINARY,
    'slemlp_kidney_binary': SLEMLP_KIDNEY_BINARY,
}
epochs = {'slemlp': 100,
          'slemlp_binary': 100,
          'slemlp_kidney_binary': 100}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=list(model_factory.keys()), type=str, help='model to run')
    # parser.add_argument('drop_prob', type=float, help='dropout probability')
    return parser.parse_args()


def main():
    args = parse_args()
    model = model_factory[args.model]()
    max_epochs = epochs[args.model]
    trainer = pl.Trainer(max_epochs=max_epochs, gpus=0, default_root_dir=f'./runs/{args.model}')
    trainer.fit(model)


if __name__ == '__main__':
    main()