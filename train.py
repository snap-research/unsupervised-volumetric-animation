"""
Copyright Snap Inc. 2023. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import matplotlib

matplotlib.use('Agg')

import os
import pickle
import sys
from argparse import ArgumentParser
from time import gmtime, strftime

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from frames_dataset import FramesDataset
from logger import VisualizationCallback
from model import UVASystem


class RegisterLoggers(Callback):

    def on_train_start(self, trainer, module):
        module.canonical_generator.log = module.log
        module.pnp_pose.log = module.log


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--log_dir", default='tb-december-log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--cache_dataset",
                        default=False,
                        action='store_true',
                        help="use cached dataset instead of iterating")

    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.full_load(f)

    experiment_name = os.path.basename(opt.config).split('.')[0]
    experiment_version = strftime("%d_%m_%y_%H.%M.%S", gmtime())

    print("Experiment name:", experiment_name)
    print("Experiment version:", experiment_version)
    num_gpus = torch.cuda.device_count()
    if 'WORLD_SIZE' in os.environ:
        num_nodes = int(os.environ['WORLD_SIZE']) // num_gpus
    else:
        num_nodes = 1
    print("Number of gpus:", num_gpus)
    print("Number of nodes:", num_nodes)

    if opt.cache_dataset:
        cache_path = config['dataset_params']['root_dir'] + '/cache.pkl'
        if os.path.exists(cache_path):
            dataset = pickle.load(open(cache_path, 'rb'))
            print("Loading dataset from cache.")
        else:
            dataset = FramesDataset(is_train=True, resolution=config['video_resolution'], **config['dataset_params'])
            pickle.dump(dataset, open(cache_path, 'wb'))
    else:
        dataset = FramesDataset(is_train=True, resolution=config['video_resolution'], **config['dataset_params'])

    train_dataloader = DataLoader(dataset,
                                  batch_size=config['batch_size'],
                                  num_workers=config['num_workers'],
                                  drop_last=True,
                                  shuffle=True)

    model = UVASystem(resolution=config['video_resolution'],
                      num_channels=config['num_channels'],
                      max_videos=dataset.max_videos,
                      model_params=config['model_params'],
                      optimizer_params=config['optimizer_params'],
                      loss_params=config['loss_params'])

    if opt.checkpoint is not None:
        model.load_state_dict(torch.load(opt.checkpoint, map_location=next(model.parameters()).device)['state_dict'])

    logger = TensorBoardLogger(opt.log_dir, name=experiment_name, version=experiment_version)
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=5 if 'checkpoint_frequency' not in config else config['checkpoint_frequency'], save_last=True)

    callbacks = [
        VisualizationCallback(dataset, **config['visualization_params']),
        RegisterLoggers(), checkpoint_callback
    ]

    trainer = Trainer(callbacks=callbacks,
                      devices=-1,
                      num_nodes=num_nodes,
                      accelerator='gpu',
                      max_epochs=config['epochs'],
                      logger=logger,
                      detect_anomaly=False)

    trainer.fit(model, train_dataloader)
