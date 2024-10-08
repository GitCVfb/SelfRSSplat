import sys
sys.path.append('softsplat_main')
from SoftSplatModel import SoftSplatBaseline

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SelfRSSR(nn.Module):
    def __init__(self):
        super(SelfRSSR, self).__init__()

        self.SelfRSSR = SoftSplatBaseline()

    def forward(self, img, bidir_optical_flow, timestep, gamma, cH, img_H): # 0 <= gamma <= 1,  -gamma/2 <= timestep <= 1 - gamma/2
        I0 = img[:,0:3,:,:].unsqueeze(2)
        I1 = img[:,3:6,:,:].unsqueeze(2)
        img_input = torch.cat([I0, I1], dim=2)#[B, 3, 2, H, W]
        #bidir_optical_flow: [2B, 2, H, W]
        
        It_p, warped_img = self.SelfRSSR(img_input, bidir_optical_flow, timestep, gamma, cH, img_H)

        return It_p, warped_img


