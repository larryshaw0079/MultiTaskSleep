"""
@Time    : 2021/9/12 22:08
@File    : main.py
@Software: PyCharm
@Desc    : 
"""

import os
import argparse
import random
import shutil
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from multitask_sleep.model import MultiTaskSleep


def setup_seed(seed):
    warnings.warn(f'You have chosen to seed ({seed}) training. This will turn on the CUDNN deterministic setting, '
                  f'which can slow down your training considerably! You may see unexpected behavior when restarting '
                  f'from checkpoints.')

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--data-path', type=str, default='/data/DataHub/EmotionRecognition/SEED/Preprocessed_EEG')
    parser.add_argument('--data-name', type=str, default='SEED',
                        choices=['SEED', 'SEED-IV', 'DEAP', 'AMIGOS', 'ISRUC', 'SLEEPEDF'])
    parser.add_argument('--task', type=str, default='stage', choices=['stage', 'apnea'], help='only valid for ISRUC')
    parser.add_argument('--modal', type=str, default='eeg', choices=['eeg', 'pps'])
    parser.add_argument('--save-path', type=str, default='./cache/tmp')
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--preprocessing', choices=['none', 'standard'], default='standard')
    parser.add_argument('--fs', type=int, default=None)

    # Model
    parser.add_argument('--input-channel-v1', type=int, default=None)
    parser.add_argument('--input-channel-v2', type=int, default=None)

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--kfold', type=int, default=10)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--lr-schedule', type=int, nargs='*', default=[120, 160])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    # Optimization
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9, help='Only valid for SGD optimizer')

    parser.add_argument('--seed', type=int, default=2020)

    args_parsed = parser.parse_args()

    if verbose:
        message = ''
        message += '-------------------------------- Args ------------------------------\n'
        for k, v in sorted(vars(args_parsed).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '-------------------------------- End ----------------------------------'
        print(message)

    return args_parsed


def main(run_id, train_patients, test_patients, args):
    print('Train patient ids:', train_patients)
    print('Test patient ids:', test_patients)

    if args.data_name == 'SEED' or args.data_name == 'SEED-IV':
        input_size = 200
    elif args.data_name == 'DEAP':
        input_size = 128
    elif args.data_name == 'AMIGOS':
        input_size = 128
    elif args.data_name == 'ISRUC':
        input_size = 6000
    elif args.data_name == 'SLEEPEDF':
        input_size = 3000
    else:
        raise ValueError

    # model = MultiTaskSleep(in_channel, mid_channel, feature_dim=16, num_apnea, num_stage, apnea_class=2, stage_class=5)
    # model.cuda(device=args.device)


if __name__ == '__main__':
    args = parse_args()

    if args.seed is not None:
        setup_seed(args.seed)

    if not os.path.exists(args.save_path):
        warnings.warn(f'The path {args.save_path} dost not existed, created...')
        os.makedirs(args.save_path)
    elif not args.resume:
        warnings.warn(f'The path {args.save_path} already exists, deleted...')
        shutil.rmtree(args.save_path)
        os.makedirs(args.save_path)

    patients = None
    if args.data_name == 'SLEEPEDF':
        patients = os.listdir(args.data_path)
        patients = np.sort(patients)
    elif args.data_name == 'ISRUC':
        patients = os.listdir(args.data_path)
        patients = np.sort(patients)
    elif args.data_name == 'DEAP':
        patients = np.arange(32)
    elif args.data_name == 'AMIGOS':
        patients = np.arange(40)
    else:
        raise ValueError

    assert args.kfold <= len(patients)
    assert args.fold < args.kfold
    kf = KFold(n_splits=args.kfold)

    for i, (train_index, test_index) in enumerate(kf.split(patients)):
        if i == args.fold:
            print(f'[INFO] Running cross validation for {i + 1}/{args.kfold} fold...')
            train_patients, test_patients = patients[train_index].tolist(), patients[test_index].tolist()
            main(i, train_patients, test_patients, args)
            break
