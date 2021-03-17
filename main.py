import argparse
import pytorch_lightning as pl

from models.sle_mlp import SLEMLP
from archive.sle_mlp_binary import SLEMLP_BINARY, SLEMLP_KIDNEY_BINARY

model_factory = {
    'slemlp': SLEMLP,
    'slemlp_binary': SLEMLP_BINARY,
    'slemlp_kidney_binary': SLEMLP_KIDNEY_BINARY,
}
epochs = {'slemlp': 200,
          'slemlp_binary': 100,
          'slemlp_kidney_binary': 100,
}

hparams = {'lr': 0.001,
           'lr_half_time': 20,
           'dropout_prob': 0.5,
           'loss_cls_weights': [0.33, 0.33, 0.33],
           'l2_strength': 1e-2,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=list(model_factory.keys()), type=str, help='model to run')
    # parser.add_argument('drop_prob', type=float, help='dropout probability')
    return parser.parse_args()


def main():
    args = parse_args()
    model = model_factory[args.model](hparams)
    max_epochs = epochs[args.model]
    trainer = pl.Trainer(max_epochs=max_epochs, gpus=0, default_root_dir=f'./runs/{args.model}')
    trainer.fit(model)


def test():
    args = parse_args()
    model = model_factory[args.model]
    ckpt_path = "runs/slemlp/lightning_logs/version_18/checkpoints/epoch=18-step=18.ckpt"
    model = model.load_from_checkpoint(ckpt_path)
    max_epochs = epochs[args.model]
    trainer = pl.Trainer(max_epochs=max_epochs, gpus=0, default_root_dir=f'./runs/{args.model}')
    trainer.test(model)

if __name__ == '__main__':
    main()
    # test()