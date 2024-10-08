import sys
sys.path.append('gmflow')
from gmflow import GMFlow

import random
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile

import os
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

from package_core.net_basics import *
from package_core.model_base import *
from package_core.losses import *
from package_core.flow_utils import *
from package_core.image_proc import *
from frame_loss import *

from network_SelfRSSR import SelfRSSR

class Model(ModelBase):
    def __init__(self, opts):
        super(Model, self).__init__()
        
        self.opts = opts
        
        # create networks
        self.model_names=['flow', 'vfi']
        
        self.net_flow = GMFlow(feature_channels=128,
                               num_scales=1,
                               upsample_factor=8,
                               num_head=1,
                               attention_type='swin',
                               ffn_dim_expansion=4,
                               num_transformer_layers=6,
                               ).cuda()
        
        self.net_vfi = SelfRSSR().cuda()
        
        self.print_networks(self.net_flow)
        self.print_networks(self.net_vfi)
        
        # load in initialized network parameters
        if opts.test_pretrained_VFI:
            self.load_pretrained_OF_model(opts.model_label, self.opts.log_dir)#load pretrained optical flow model directly
            self.load_pretrained_GS_model(opts.model_label, self.opts.log_dir)#load pretrained GS-based VFI model directly
        elif not opts.is_training or opts.continue_train:
            self.load_checkpoint(opts.model_label)
        else:
            self.load_pretrained_OF_model(self.opts.model_label_pretrained_GS, self.opts.log_dir_pretrained_GS)#load pretrained optical flow model directly
            self.load_pretrained_GS_model(self.opts.model_label_pretrained_GS, self.opts.log_dir_pretrained_GS)#load pretrained GS-based VFI model
        
        
        if self.opts.is_training:
            # initialize optimizers
            
            self.optimizer_G = torch.optim.Adam([{'params': self.net_vfi.parameters()}], lr=opts.lr)
            #self.optimizer_G = torch.optim.Adam([{'params': self.net_flow.parameters(), 'lr': 1e-5}, {'params': self.net_vfi.parameters()}], lr=opts.lr)
            
            self.optimizer_names = ['G']
            self.build_lr_scheduler()
            
            # create losses
            self.loss_fn_perceptual = PerceptualLoss(loss=nn.L1Loss())
            self.loss_fn_L1 = L1Loss()
            self.census = Ternary()
            
            self.loss_fn_tv2 = VariationLoss(nc=2)
            
            ###Initializing VGG16 model for perceptual loss
            self.MSE_LossFn = nn.MSELoss()
            vgg16 = torchvision.models.vgg16(pretrained=True)
            self.vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
            self.vgg16_conv_4_3.to('cuda')
            for param in self.vgg16_conv_4_3.parameters():
                param.requires_grad = False

    def set_input(self, _input):
        im_rs, im_gs, im_gs_f, cH = _input
        self.im_rs = im_rs.cuda()
        self.im_gs = im_gs
        self.im_gs_f = im_gs_f
        self.cH = cH

        if self.im_gs is not None:
            self.im_gs = self.im_gs.cuda()
        if self.im_gs_f is not None:
            self.im_gs_f = self.im_gs_f.cuda()

    #For testing
    def GS_syn(self, time, gamma):# -gamma/2 <= time <= 1 - gamma/2, 0 <= gamma <= 1
        _, _, H, W = self.im_rs.size()
        im_rs0 = self.im_rs[:,0:3,:,:].clone()
        im_rs1 = self.im_rs[:,3:6,:,:].clone()

        results_dict = self.net_flow(im_rs0, im_rs1,
                                 attn_splits_list=[2],
                                 corr_radius_list=[-1],
                                 prop_radius_list=[-1],
                                 pred_bidir_flow=True,
                                 )

        flow_bidir = results_dict['flow_preds'][-1]  # [2*B, 2, H, W]
        
        gs_t_final, warped_img = self.net_vfi(self.im_rs[:,:6,:,:], flow_bidir, time, gamma, 0, H)#testing
        
        return gs_t_final
    
    def GS_syn_2(self, time, gamma):# -gamma/2 <= time <= 1 - gamma/2, 0 <= gamma <= 1
        _, _, H, W = self.im_rs.size()
        im_rs0 = self.im_rs[:,0:3,:,:].clone()
        im_rs1 = self.im_rs[:,3:6,:,:].clone()
        im_rs2 = self.im_rs[:,6:9,:,:].clone()

        results_dict = self.net_flow(im_rs0, im_rs1,
                                 attn_splits_list=[2],
                                 corr_radius_list=[-1],
                                 prop_radius_list=[-1],
                                 pred_bidir_flow=True,
                                 )
        flow_bidir = results_dict['flow_preds'][-1]  # [2*B, 2, H, W]
        gs_t,_ = self.net_vfi(torch.cat([im_rs0, im_rs1], dim=1), flow_bidir, time, gamma, 0, H)#testing
        
        results_dict = self.net_flow(im_rs1, im_rs2,
                                 attn_splits_list=[2],
                                 corr_radius_list=[-1],
                                 prop_radius_list=[-1],
                                 pred_bidir_flow=True,
                                 )
        flow_bidir = results_dict['flow_preds'][-1]  # [2*B, 2, H, W]
        gs_t_plus_1,_ = self.net_vfi(torch.cat([im_rs1, im_rs2], dim=1), flow_bidir, time, gamma, 0, H)#testing
        
        results_dict = self.net_flow(gs_t_plus_1, gs_t,
                                 attn_splits_list=[2],
                                 corr_radius_list=[-1],
                                 prop_radius_list=[-1],
                                 pred_bidir_flow=True,
                                 )
        flow_bidir = results_dict['flow_preds'][-1]  # [2*B, 2, H, W]
        rs_1_pred,_ = self.net_vfi(torch.cat([gs_t_plus_1, gs_t], dim=1), flow_bidir, time, gamma, 0, H)#testing
        
        return gs_t
        
    #For training
    def forward(self, im_0, im_1, time, gamma):# -gamma/2 <= time <= 1 - gamma/2, 0 <= gamma <= 1
        results_dict = self.net_flow(im_0, im_1,
                                 attn_splits_list=[2],
                                 corr_radius_list=[-1],
                                 prop_radius_list=[-1],
                                 pred_bidir_flow=True,
                                 )

        flow_bidir = results_dict['flow_preds'][-1]  # [2*B, 2, H, W]
        
        im_t_pred, warped_img = self.net_vfi(torch.cat([im_0, im_1], dim=1), flow_bidir, time, gamma, self.cH, self.opts.img_H)#training
        
        if self.opts.is_training:
            return im_t_pred, warped_img
        
        return im_t_pred


    def optimize_parameters(self):
        
        #===========================================================#
        #                   Initialize losses                       #
        #===========================================================#
        self.loss_L1 = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_perceptual = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_census = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_warp = torch.tensor([0.], requires_grad=True).cuda().float()
        #self.loss_flow_smoothness = torch.tensor([0.], requires_grad=True).cuda().float()

        B,_,_,_ = self.im_rs.size()
        im_gt = self.im_rs[:,3:6,:,:].clone()

        timestep = float(torch.rand(1) - self.opts.gamma/2) #train RSSR for Carla-RS and Fastec-RS   #-gamma/2 <= time <= 1 - gamma/2
        if self.opts.dataset_type=='BSRSC':
            timestep = float(1.0 - self.opts.gamma  )#train RSSR for BS-RSC
        
        #RS -> GS
        gs_t, warped_gs_t               = self.forward(self.im_rs[:,0:3,:,:], self.im_rs[:,3:6,:,:], timestep, self.opts.gamma)
        gs_t_plus_1, warped_gs_t_plus_1 = self.forward(self.im_rs[:,3:6,:,:], self.im_rs[:,6:9,:,:], timestep, self.opts.gamma)

        #GS -> RS
        rs_1_pred, warped_rs_1   = self.forward(gs_t_plus_1, gs_t, timestep, self.opts.gamma)
        
        self.loss_L1 += self.charbonier_loss(rs_1_pred, im_gt, mean=True) * self.opts.lamda_L1 *10.0#Charbonnier
        self.loss_perceptual += self.loss_fn_perceptual.get_loss(rs_1_pred, im_gt) * self.opts.lamda_perceptual
        
        if timestep < (0.5 - self.opts.gamma/2):
            self.loss_warp += self.charbonier_loss(warped_rs_1[:B,], im_gt, mean=True) * self.opts.lamda_L1 *10.0#color consistency
        else:
            self.loss_warp += self.charbonier_loss(warped_rs_1[B:,], im_gt, mean=True) * self.opts.lamda_L1 *10.0
        
        # sum them up
        self.loss_G = self.loss_L1 + self.loss_perceptual + self.loss_warp
                        
        # Optimize 
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

    def warp(self, img, flow):
        B, _, H, W = flow.shape
        xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
        yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
        grid = torch.cat([xx, yy], 1).to(img)
        flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
        grid_ = (grid + flow_).permute(0, 2, 3, 1)
        output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
        return output

    def charbonier_loss(self, pred_im, im_gt, epsilon=0.001, mean=True):
        x = pred_im - im_gt
        loss = torch.mean(torch.sqrt(x ** 2 + epsilon ** 2))
        return loss
    
    # save networks to file 
    def save_checkpoint(self, label):
        self.save_network(self.net_flow, 'flow', label, self.opts.log_dir)
        self.save_network(self.net_vfi,  'vfi',  label, self.opts.log_dir)

    def load_checkpoint(self, label):
        self.load_network(self.net_flow, 'flow', label, self.opts.log_dir)
        self.load_network(self.net_vfi,  'vfi',  label, self.opts.log_dir)

    def load_pretrained_GS_model(self, label, save_dir):
        def convert1(param):
            return {
            "SelfRSSR."+k: v
                for k, v in param.items()
            }
        def convert2(param):
            return {
            "module.SelfRSSR."+k: v
                for k, v in param.items()
            }
        
        save_filename = '%s_net_%s.pth' % (label, 'vfi')
        save_path = os.path.join(save_dir, save_filename)
        print('load model from ', save_path)

        checkpoint = torch.load(save_path)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        self.net_vfi.load_state_dict(convert1(weights), False)#For single-GPU training
        #self.net_vfi.load_state_dict(convert2(weights), False)#For multip-GPU training

    def load_pretrained_OF_model(self, label, save_dir):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }       
        def convert1(param):
            return {
            "module."+k: v
                for k, v in param.items()
            }
        save_filename = '%s_net_%s.pth' % (label, 'flow')
        save_path = os.path.join(save_dir, save_filename)
        print('load model from ', save_path)
        
        checkpoint = torch.load(save_path)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        self.net_flow.load_state_dict(weights)#For single-GPU training
        #self.net_flow.load_state_dict(convert1(weights))#For multip-GPU training

    def get_current_scalars(self):
        losses = {}
        losses['loss_G'] = self.loss_G.item()
        losses['loss_L1'] = self.loss_L1.item()
        losses['loss_percep'] = self.loss_perceptual.item()
        #losses['loss_census'] = self.loss_census.item()
        losses['loss_warp'] = self.loss_warp.item()
        return losses

    def get_current_visuals(self):
        output_visuals = {}

        output_visuals['im_rs'] = self.im_rs[:,-3:,:,:].clone()
        
        return output_visuals
