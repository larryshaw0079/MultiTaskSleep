"""
@Time    : 2021/9/12 22:11
@File    : dataset.py
@Software: PyCharm
@Desc    : 
"""
import os
from typing import List

import numpy as np
import torch

from torch.utils.data import Dataset


class ISRUCDataset(Dataset):
    num_subject = 99
    fs = 200

    def __init__(self, data_path, num_epoch, transform=None, patients: List = None, modal='eeg', return_idx=False,
                 verbose=True, standardize='none', task='stage'):
        assert isinstance(patients, list)

        self.data_path = data_path
        self.transform = transform
        self.patients = patients
        self.modal = modal
        self.return_idx = return_idx

        assert modal in ['eeg', 'pps', 'all']
        assert task in ['stage', 'apnea']

        self.data = []
        self.labels = []

        for i, patient in enumerate(patients):
            data = np.load(os.path.join(data_path, patient))
            if modal == 'eeg':
                recordings = np.stack([data['F3_A2'], data['C3_A2'], data['F4_A1'], data['C4_A1'],
                                       data['O1_A2'], data['O2_A1']], axis=1)
            elif modal == 'pps':
                recordings = np.stack([data['X1'], data['X2'], data['X3'], data['LOC_A2'], data['ROC_A1']], axis=1)
            elif modal == 'all':
                recordings = np.stack([data['F3_A2'], data['C3_A2'], data['F4_A1'], data['C4_A1'],
                                       data['O1_A2'], data['O2_A1'], data['X1'], data['X2'], data['X3'],
                                       data['LOC_A2'], data['ROC_A1']], axis=1)
            else:
                raise ValueError

            if task == 'stage':
                annotations = data['stage_label'].flatten()
            elif task == 'apnea':
                annotations = data['apnea_label'].flatten()
            else:
                raise ValueError

            if standardize == 'none':
                pass
            elif standardize == 'standard':
                recordings_min = np.expand_dims(recordings.min(axis=-1), axis=-1)
                recordings_max = np.expand_dims(recordings.max(axis=-1), axis=-1)
                recordings = (recordings - recordings_min) / (recordings_max - recordings_min)
            else:
                raise ValueError

            if verbose:
                print(
                    f'[INFO] Processing the {i + 1}-th patient {patient} [shape: {recordings.shape} - {annotations.shape}] ...')

            recordings = recordings[:(recordings.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch,
                                                                                             *recordings.shape[1:])
            annotations = annotations[:(annotations.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch)

            assert recordings.shape[:2] == annotations.shape[:2], f'{patient}: {recordings.shape} - {annotations.shape}'

            self.data.append(recordings)
            self.labels.append(annotations)

        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)
        self.idx = np.arange(self.data.shape[0] * self.data.shape[1]).reshape(-1, self.data.shape[1])
        self.full_shape = self.data[0].shape

    def __getitem__(self, item):
        x = self.data[item]
        y = self.labels[item]

        x = x.astype(np.float32)
        y = y.astype(np.long)

        if self.transform is not None:
            x = self.transform(x)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        if self.return_idx:
            return x, y, torch.from_numpy(self.idx[item].astype(np.long))
        else:
            return x, y

    def __len__(self):
        return len(self.data)
