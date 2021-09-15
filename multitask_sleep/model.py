"""
@Time    : 2021/9/12 22:11
@File    : model.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn

from .backbone import R1DNet


class MultiTaskSleep(nn.Module):
    def __init__(self, in_channel, mid_channel, feature_dim, num_apnea, num_stage, apnea_class=2, stage_class=5):
        super().__init__()

        self.feature_extractor = R1DNet(in_channel=in_channel, mid_channel=mid_channel, feature_dim=feature_dim,
                                        final_fc=False)

        self.sleep_apnea_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(mid_channel * 16, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )

        self.sleep_apnea_header = nn.Sequential(
            nn.Linear(feature_dim + stage_class, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, apnea_class)
        )

        self.sleep_stage_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(mid_channel * 16, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )

        self.sleep_stage_header = nn.Sequential(
            nn.Linear(feature_dim + apnea_class, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, stage_class)
        )

        self.register_buffer("apnea_memory", torch.randn(num_apnea, apnea_class))
        self.register_buffer("stage_memory", torch.randn(num_stage, stage_class))

    @torch.no_grad()
    def __update_memory(self, y1, y2, idx):
        self.apnea_memory[idx] = y1
        self.stage_memory[idx] = y2

    def forward(self, x, idx):
        x = self.feature_extractor(x)
        z1 = self.sleep_apnea_branch(x)
        z2 = self.sleep_stage_branch(x)

        y1 = self.sleep_apnea_header(torch.cat([z1, self.stage_memory[idx]], dim=-1))
        y2 = self.sleep_stage_header(torch.cat([z2, self.apnea_memory[idx]], dim=-1))

        self.__update_memory(y1, y2, idx)

        return y1, y2
