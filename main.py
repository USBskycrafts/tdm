import argparse
import os

import yaml
from pytorch_lightning import Trainer

from tdm.dataset.interface import Dataset
from tdm.models.ddpm import DDPM
from tdm.utils import load_instance, load_instances


def build_argparser():
    parser = argparse.ArgumentParser(description="A script to process data.")
    parser.add_argument('--gpus', type=str, default='0', help='GPU ids to use')
    parser.add_argument('--config', type=str,
                        default='config.yaml', help='Path to the config file')
    parser.add_argument('--resume', type=str,
                        default=None, help='Path to the checkpoint to resume from')
    parser.add_argument('--test-only', type=bool, default=False,
                        help='Whether to only test the model')
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    # Set the GPU ids
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # Load the config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create the model
    model = load_instance(config['model'])
    dataset = Dataset(config['dataset'])


    # create trainer
    trainer = Trainer(
        logger=load_instance(config['trainer']['logger']),
        callbacks=load_instances(config['trainer']['callbacks']),
        max_epochs=config['trainer']['max_epochs'],
        precision='16-mixed'
    )

    if not args.test_only:
        ckpt_path = None
        if args.resume:
            ckpt_path = args.resume 
        # train the model
        trainer.fit(model, dataset, ckpt_path=ckpt_path)
            

    # test the model
    trainer.test(model, dataset)


if __name__ == "__main__":
    main()
