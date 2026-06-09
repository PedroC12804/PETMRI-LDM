import torch
import torch.nn as nn


class MRIEncoder(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()

        # 256 -> 128 -> 64 -> 32 -> 16
        self.stage1 = nn.Sequential(

            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(),

            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
        )

        # 16 -> 8
        self.stage2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
        )

        # 8 -> 4
        self.stage3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
        )


    def forward(self, x):
        s16 = self.stage1(x)  # [B,256,16,16]

        s8 = self.stage2(s16)  # [B,512,8,8]

        s4 = self.stage3(s8)  # [B,512,4,4]


        return {
            "16": s16,
            "8": s8,
            "4": s4,
        }