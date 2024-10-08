import torch
import torch.nn as nn

from softsplat import Softsplat
from GridNet import GridNet
from torch.nn.functional import interpolate, grid_sample
from einops import repeat


def generate_2D_grid(H, W):
    x = torch.arange(0, W, 1).float().cuda() 
    y = torch.arange(0, H, 1).float().cuda()

    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)
    
    grid = torch.stack([xx, yy], dim=0) 
    return grid

def GS_temporal_offset(t, grid_rows, H, gamma=1.0):# 0 <= t <= 1,  -gamma/2 <= t <= 1 - gamma/2
    tau = t + gamma - gamma / H * grid_rows + 0.0001
    
    return tau

def interflow_syn(bidir_optical_flow, t, gamma, cH, crop_sz_H, img_H, scale):# -gamma/2 <= tau <= 1 - gamma/2
    B, _, _, W = bidir_optical_flow.size()#flow: [B, 4, H, W]
    

    H = int(img_H * scale)
    grid_rows = generate_2D_grid(H, W)[1]
    grid_rows = grid_rows.unsqueeze(0).unsqueeze(0)
    grid_rows = torch.cat([grid_rows] * B, dim=0)
    
    tau_y_forward_ = GS_temporal_offset(t, grid_rows, H, gamma)
    tau_y_forward  = tau_y_forward_[:,:,int(cH*scale):int((cH+crop_sz_H)*scale), :]

    denom_forward  = 1.0 + gamma * ( bidir_optical_flow[:,1].unsqueeze(1)) / H
    denom_backward = 1.0 + gamma * (-bidir_optical_flow[:,3].unsqueeze(1)) / H

    factor_left_forward = tau_y_forward / denom_forward
    factor_right_backward = (1.0 - tau_y_forward) / denom_backward

    flow_0_t_from_forward = factor_left_forward * bidir_optical_flow[:,0:2]
    flow_1_t_from_backward = factor_right_backward * bidir_optical_flow[:,2:4]

    return flow_0_t_from_forward, flow_1_t_from_backward


# convert [0, 1] to [-1, 1]
def preprocess(x):
    return x * 2 - 1

# convert [-1, 1] to [0, 1]
def postprocess(x):
    return torch.clamp((x + 1) / 2, 0, 1)


class BackWarp(nn.Module):
    def __init__(self, clip=True):
        super(BackWarp, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip = clip

    def forward(self, img, flow):
        b, c, h, w = img.shape
        gridY, gridX = torch.meshgrid(torch.arange(h), torch.arange(w))
        gridX, gridY = gridX.to(self.device), gridY.to(self.device)

        u = flow[:, 0]  # W
        v = flow[:, 1]  # H

        x = repeat(gridX, 'h w -> b h w', b=b).float() + u
        y = repeat(gridY, 'h w -> b h w', b=b).float() + v

        # normalize
        x = (x / w) * 2 - 1
        y = (y / h) * 2 - 1

        # stacking X and Y
        grid = torch.stack((x, y), dim=-1)

        # Sample pixels using bilinear interpolation.
        if self.clip:
            output = grid_sample(img, grid, mode='bilinear', align_corners=True, padding_mode='border')
        else:
            output = grid_sample(img, grid, mode='bilinear', align_corners=True)
        return output


class SoftSplatBaseline(nn.Module):
    def __init__(self, predefined_z=False, act=nn.PReLU):
        super(SoftSplatBaseline, self).__init__()

        self.fwarp = Softsplat()
        self.bwarp = BackWarp(clip=False)
        self.feature_pyramid = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                act(),
                nn.Conv2d(32, 32, 3, 1, 1),
                act()
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, 2, 1),
                act(),
                nn.Conv2d(64, 64, 3, 1, 1),
                act()
            ),
            nn.Sequential(
                nn.Conv2d(64, 96, 3, 2, 1),
                act(),
                nn.Conv2d(96, 96, 3, 1, 1),
                act()
            ),
        ])
        self.synth_net = GridNet(dim=32, act=act)
        self.predefined_z = predefined_z
        if predefined_z:
            self.alpha = nn.Parameter(-torch.ones(1))
        else:
            self.v_net = nn.Sequential(
                nn.Conv2d(6, 64, 3, 1, 1),
                act(),
                nn.Conv2d(64, 64, 3, 1, 1),
                act(),
                nn.Conv2d(64, 1, 3, 1, 1)
            )

    def forward(self, x, flow, target_t, gamma, cH, img_H):# 0 <= target_t <= 1,  -gamma/2 <= t <= 1 - gamma/2
        img_input = x
        #flow: [2B, 2, H, W]
        x = preprocess(x)#[B, 3, 2, H, W]
        b = x.shape[0]
        ##target_t = target_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fr0, fr1 = x[:, :, 0], x[:, :, 1]#[B, 3, H, W]
        
        f_lv = torch.cat([fr0, fr1], dim=0)
        pyramid = [f_lv]
        for feat_extractor_lv in self.feature_pyramid:
            f_lv = feat_extractor_lv(f_lv)
            pyramid.append(f_lv)

        # Z importance metric
        brightness_diff = torch.abs(self.bwarp(torch.cat([fr1, fr0], dim=0), flow) - torch.cat([fr0, fr1], dim=0))
        if self.predefined_z:
            z = self.alpha * torch.sum(brightness_diff, dim=1, keepdim=True)
        else:
            z = self.v_net(torch.cat([torch.cat([fr0, fr1]), -brightness_diff], dim=1))

        # warping
        n_lv = len(pyramid)
        warped_feat_pyramid = []
        for lv in range(n_lv):
            f_lv = pyramid[lv]
            scale_factor = f_lv.shape[-1] / flow.shape[-1]
            flow_lv = interpolate(flow, scale_factor=scale_factor, mode='bilinear', align_corners=False) * scale_factor
            flow01, flow10 = torch.split(flow_lv, b, dim=0)

            ##flow0t, flow1t = flow01 * target_t, flow10 * (1 - target_t)
            flow0t, flow1t = interflow_syn(torch.cat((flow01, flow10), 1), target_t, gamma, cH, x.shape[3], img_H, scale_factor)

            flowt = torch.cat([flow0t, flow1t], dim=0)
            z_lv = interpolate(z, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            warped_f_lv = self.fwarp(f_lv, flowt, z_lv)
            warped_feat_pyramid.append(warped_f_lv)
            
            if lv==0:
                warped_img_lv0 = self.fwarp(torch.cat((img_input[:,:,0,:,:], img_input[:,:,1,:,:]), 0), flowt, z_lv)#[2B,3,H,W]

        concat_warped_feat_pyramid = []
        for feat_lv in warped_feat_pyramid:
            feat0_lv, feat1_lv = feat_lv.chunk(2, dim=0)
            feat_lv = torch.cat([feat0_lv, feat1_lv], dim=1)
            concat_warped_feat_pyramid.append(feat_lv)
        output = self.synth_net(concat_warped_feat_pyramid)
        return postprocess(output), warped_img_lv0

