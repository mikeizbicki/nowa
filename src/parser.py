from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate.'
)
parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
)
parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
)
parser.add_argument(
        '--input_data_dir',
        type=str,
        default='data',
        help='Directory to put the input data.'
)
parser.add_argument(
        '--log_dir_out',
        type=str,
        default='log/local',
        help='Directory to put the log data.'
)
parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
)
parser.add_argument(
        '--numproc',
        type=int,
        default=1
        )
parser.add_argument(
        '--procid',
        type=int,
        default=0
        )
parser.add_argument(
        '--seed',
        type=int,
        default=0
        )
parser.add_argument(
        '--model',
        type=str,
        default=''
        )
parser.add_argument(
        '--dataset',
        type=str,
        default=''
        )
parser.add_argument(
        '--same_seed',
        default=False,
        action='store_true'
        )
parser.add_argument(
        '--maxcpu',
        type=int,
        default=0
        )
parser.add_argument(
        '--induced_bias',
        type=float,
        default=0
        )
parser.add_argument(
        '--allcheckpoints',
        default=False,
        action='store_true'
        )

FLAGS, unparsed = parser.parse_known_args()
if unparsed:
    print('unparsed arguments: ',unparsed)
    exit(1)

