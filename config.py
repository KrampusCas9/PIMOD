import os
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn


optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}

class Config(object):
    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                setattr(self, key, value)

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):


    parser = argparse.ArgumentParser()

    # Train
    time_now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    parser.add_argument('--time', type=str, default=f"{time_now}")
    parser.add_argument('--name', type=str, default=f"{time_now}")
    # Make sure the result path exists
    parser.add_argument('--result_path', type=str, default="./Exp")
    parser.add_argument('--memo', type=str, default="None")

    # Data parameters
    parser.add_argument('--model', type=str, default="Model_LSTM")
    parser.add_argument('--finetune_id', type=int, default=0)

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epoch', type=int, default=30)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--cls_weight', type=float, default=1)
    parser.add_argument('--learning_rate', type=float, default=3e-3)
    parser.add_argument('--freeze', type=bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='Adam')

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)