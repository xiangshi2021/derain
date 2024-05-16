# -*- coding: utf-8 -*-
"""
arch_modele （通用模块）

"""

import torch
import torch.nn as nn
import numpy as np
import os
import time
from torch.optim.lr_scheduler import MultiStepLR
from Loss.loss import PerceptualLoss, EDGELoss
from Loss.SSIM import SSIM
from models.hat_arch import HAT
from func import get_mask, torch_variable
from data.dataset_util_stage2 import RainDataset
from torch.utils.data import DataLoader

class trainer:
    def __init__(self, opt):
        self.net_H = HAT(overlap_ratio=0,upscale=1,upsampler='pixelshuffle').cuda()
        self.net_H  = torch.nn.DataParallel(self.net_H)
        print('# generator parameters:', sum(param.numel() for param in self.net_H.parameters()))
        self.optim1 = torch.optim.Adam(self.net_H.parameters(), lr=opt.lr)
        self.sche = MultiStepLR(self.optim1, milestones=[30, 50, 80, 120], gamma=0.2)
        self.iter = opt.iter
        self.batch_size = opt.batch_size
        self.out_path = opt.result
        print('Loading dataset ...\n')

        train_dataset = RainDataset(opt) #list成对返回雨、无雨
        valid_dataset = RainDataset(opt, is_eval=True)
        train_size = len(train_dataset)
        valid_size = len(valid_dataset)

        self.train_loader = DataLoader(train_dataset, num_workers=26, batch_size=opt.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, num_workers=26,batch_size=opt.batch_size)

        print("# train set : {}".format(train_size))
        print("# eval set : {}".format(valid_size))
        
        #losses
        #ssim loss
        self.ssim = SSIM().cuda()
        #AAttention loss
        self.criterionATT = nn.MSELoss().cuda()
        #Perceptual Loss
        self.criterionPL = PerceptualLoss()
        #MSE loss
        self.criterionMSE = nn.MSELoss().cuda()
        #EDGE loss
        self.criterionEDGE = EDGELoss()

    def forward_process(self,I, GT, is_train=True):
        M_ = []
        for i in range(I.shape[0]):
            M_.append(get_mask(np.array(I[i]),np.array(GT[i])))  # #bs中每个图求mask，get_mask在func.py中
        M_ = np.array(M_)
        M_ = torch_variable(M_, is_train)
        
        I_ = torch_variable(I, is_train)
        GT_ = torch_variable(GT, is_train)

        if is_train:
            self.net_H.train()
            O_, Attention= self.net_H(I_,)
            #losses
            loss_MSE =self.criterionMSE(O_ ,GT_.detach_())
            loss_ATT = self.criterionATT(Attention,M_.detach_())
            #preceptual loss
            loss_PL = self.criterionPL(O_, GT_.detach_())
            #ssim loss
            ssim_loss = 1-self.ssim(O_,GT_.detach_())
            #edge loss
            edge_loss = self.criterionEDGE(O_,GT_.detach_())
            #finally loss
            loss_H = loss_MSE + ssim_loss + 0.1*edge_loss + 0.5*loss_ATT + loss_PL
            output = [loss_H, O_, loss_MSE, ssim_loss, edge_loss, loss_ATT, loss_PL]
            
        else:
            self.net_.eval()
            O_ = self.net_H(I_)
            output = O_
            self.net_H.train()
        return output
    
    def train_start(self):
        count = 0
        for epoch in range(0, self.iter+1):   #self.iter=200
            f=open('train_process.txt','a')
            since = time.time()
            self.sche.step()
            lr = self.optim1.param_groups[0]['lr']
            print('%d_epochGGlearning rate = %.7f' % (epoch,lr))

            for i, data in enumerate(self.train_loader):
                count+=1
                I_, GT_ = data
                loss_G, O_, loss_MSE, ssim_loss, edge_loss, loss_ATT,loss_PL = self.forward_process(I_,GT_)
                self.optim1.zero_grad()
                loss_G.backward()
                self.optim1.step()
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                           time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
            strattime = str(epoch)

            f.write(strattime)
            f.write('      ')
            f.write('loss_MSE')
            f.write('      ')
            mesloss = str(loss_MSE.item())
            f.write(mesloss)
            f.write('      ')
            f.write('ssim_loss')
            f.write('      ')
            ssimloss = str(ssim_loss.item())
            f.write(ssimloss)
            f.write('      ')

            f.write('edge_loss')
            f.write('      ')
            edgeloss = str(edge_loss.item())
            f.write(edgeloss)
            f.write('      ')

            f.write('loss_ATT')
            f.write('      ')
            lossATT = str(loss_ATT.item())
            f.write(lossATT)
            f.write('      ')

            f.write('loss_PL')
            f.write('      ')
            lossPL = str(loss_PL.item())
            f.write(lossPL)
            f.write('\r\n')
            f.close()
            if epoch % 5 ==0:

                where_to_save = self.out_path
                where_to_save_epoch = os.path.join(where_to_save, "epoch" + str(epoch) + '/') 
                if not os.path.exists(where_to_save_epoch):
                    os.makedirs(where_to_save_epoch)
                file_name = os.path.join(where_to_save_epoch, 'hat_para.pth')
                torch.save(self.net_H.state_dict(), file_name,  _use_new_zipfile_serialization=False)


        return