import torch
import torch.nn as nn
import torchvision
import cv2
from torch.autograd import Variable
from torchvision.models.vgg import vgg16
import numpy as np
import torch.nn.functional as F
#原始的ganloss
#class GANLoss(nn.Module):
#    def __init__(self, real_label=1.0, fake_label=0.0):
#        super(GANLoss, self).__init__()
#        self.real_label = real_label
#        self.fake_label = fake_label
#        # self.loss = nn.MSELoss().cuda()
#        self.loss = nn.BCELoss().cuda()
#    def convert_tensor(self, input, is_real):
#        if is_real:
#            return Variable(torch.FloatTensor(input.size()).fill_(self.real_label)).cuda()
#        else:
#            return Variable(torch.FloatTensor(input.size()).fill_(self.fake_label)).cuda() 
#    def __call__(self, input, is_real):
#        return self.loss(input, self.convert_tensor(input,is_real).cuda())

#relative gan loss
class GANLoss(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label_val = real_label
        self.fake_label_val = fake_label
        self.loss = nn.BCEWithLogitsLoss()
        
    def get_target_label(self, input, is_real):
        if is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, is_real):
        target_label = self.get_target_label(input, is_real)
        loss = self.loss(input, target_label)
        return loss

class AttentionLoss(nn.Module):
    def __init__(self, theta=0.8, iteration=4):
        super(AttentionLoss, self).__init__()
        self.theta = theta
        self.iteration = iteration
        self.loss = nn.MSELoss().cuda()

    def __call__(self, A_, M_):
        loss_ATT = None
        for i in range(1, self.iteration+1):
            if i == 1:
                loss_ATT = pow(self.theta, float(self.iteration-i)) * self.loss(A_[i-1],M_)
            else:
                loss_ATT += pow(self.theta, float(self.iteration-i)) * self.loss(A_[i-1],M_)
        return loss_ATT

# VGG16 pretrained on Imagenet
def trainable(net, trainable):
    for param in net.parameters():
        param.requires_grad = trainable

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.model = (vgg16(pretrained = True).cuda())
        trainable(self.model, False)

        self.loss = nn.MSELoss().cuda()
        self.vgg_layers = self.model.features[:23]
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
    def get_layer_output(self,x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output

    def __call__(self, O_, T_):
        o = self.get_layer_output(O_)
        t = self.get_layer_output(T_)
        loss_PL = None
#        print("len(t)",t[0].shape)  #torch.Size([bs, 64, 128, 128])
        for i in range(len(t)):
            if i == 0:
                loss_PL = self.loss(o[i],t[i]) / float(len(t))    #相当于求4个抽取出来的特征求平均
            else:
                loss_PL += self.loss(o[i],t[i]) / float(len(t))
        return loss_PL
        
class MultiscaleLoss(nn.Module):
    def __init__(self, ld=[0.6,0.8,1.0],batch=1):
        super(MultiscaleLoss, self).__init__()
        self.loss = nn.MSELoss().cuda()
        self.ld = ld
        self.batch=batch
    def __call__(self, S_, gt):
        #1,128,256,3
        T_ = []
#        print("[0].shape[0]",S_[0].shape[0])   #bs
        for i in range(S_[0].shape[0]):
            temp = []
            x = (np.array(gt[i])*255.).astype(np.uint8)
            # print (x.shape, x.dtype)
            t = cv2.resize(x, None, fx=1.0/4.0,fy=1.0/4.0, interpolation=cv2.INTER_AREA)
            t = np.expand_dims((t/255.).astype(np.float32).transpose(2,0,1),axis=0)
#            print ("shape1",t.shape)  (1, 3, 32, 32)
            temp.append(t)
            t = cv2.resize(x, None, fx=1.0/2.0,fy=1.0/2.0, interpolation=cv2.INTER_AREA)
            t = np.expand_dims((t/255.).astype(np.float32).transpose(2,0,1),axis=0)
#            print ("shape2",t.shape)  (1, 3, 64, 64)
            temp.append(t)
            x = np.expand_dims((x/255.).astype(np.float32).transpose(2,0,1),axis=0)
#            print ("shape3",x.shape)  (1, 3, 128, 128)
            temp.append(x)   #bs中每个图为一个temp
            T_.append(temp)
#        print("lenT",T_[0][0].shape)  #第0个图中的第一个(1, 3, 32, 32)
        temp_T = []
        for i in range(len(self.ld)):   #self.ld=3
            # if self.batch == 1:
            #     temp_T.append(Variable(torch.from_numpy(T_[0][i])).cuda())
            # else:
            for j in range((S_[0].shape[0])): #bs
                if j == 0:
                    x = T_[j][i]
#                    print("x",x.shape)
                else:
                    x = np.concatenate((x, T_[j][i]), axis=0)
#                    print("xx",x.shape)
            temp_T.append(Variable(torch.from_numpy(x)).cuda())
#            print("temp_T%d"%i,temp_T[i].shape)
#            temp_T0 torch.Size([2, 3, 32, 32])   2个bs
#            temp_T1 torch.Size([2, 3, 64, 64])
#            temp_T2 torch.Size([2, 3, 128, 128])
        T_ = temp_T
        loss_ML = None
        for i in range(len(self.ld)):
            if i == 0: 
#                print("S_[i]",S_[i].shape,T_[i].shape)
                loss_ML = self.ld[i] * self.loss(S_[i], T_[i])
            else:
                loss_ML += self.ld[i] * self.loss(S_[i], T_[i])
        
        return loss_ML/float(S_[0].shape[0])

class MAPLoss(nn.Module):
    def __init__(self, gamma=0.05):
        super(MAPLoss, self).__init__()
        self.loss = nn.MSELoss().cuda()
        self.gamma = gamma

    # D_map_O, D_map_R
    def __call__(self, D_O, D_R, A_N):
        Z = Variable(torch.zeros(D_R.shape)).cuda()
        D_A = self.loss(D_O,A_N)
        D_Z = self.loss(D_R,Z)
        return self.gamma * (D_A + D_Z)
    
    
    
class Laplacian_edge(nn.Module):
    def __init__(self):
        super(Laplacian_edge, self).__init__()
#        kernel = [[-1/255, -1/255, -1/255],
#                  [-1/255, 8/255, -1/255],
#                  [-1/255, -1/255, -1/255]]
        kernel = [[-1, -1, -1],
                  [-1, 8, -1],
                  [-1, -1, -1]]
        
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).cuda()
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)
        self.weight.data = self.weight.data.cuda()   
        #只能将self.weight.data转到GPU上，而不能将self.weight整体转到GPU上
        #https://www.jianshu.com/p/749439fb026d
 
    def forward(self, x):
        
        x1 = x[:, 0 , :, :]
        x2 = x[:, 1 , :, :]
        x3 = x[:, 2 , :, :]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=1)
#        rint("x12",x1.shape)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=1)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=1)
        x = torch.cat([x1, x2, x3], dim=1)
        return x
    
    
class EDGELoss(nn.Module):
    def __init__(self):
        super(EDGELoss, self).__init__()
#        self.theta = theta
#        self.iteration = iteration
        self.edge = Laplacian_edge()
        self.loss = nn.MSELoss().cuda()
        #这里MSELoss求的为什么是差的和？估计是和Laplacian_edge中的F.conv2d有关系？
#        因此按照定义写了MSELoss，或者用求出来的值除以bs*c*h*w
#        self.loss1 = nn.MSELoss(size_average = False).cuda()
        self.relu = nn.ReLU().cuda()
    def __call__(self, derain, gt):
#        print("derain",derain.device)
#        print("gt",gt.device)
        derain_edge = self.relu(self.edge(derain))
        gt_edge = self.relu(self.edge(gt))
        
        out1 = (derain_edge-gt_edge)
        out1 =out1.pow(2)
        out1 = out1.mean()
#        print("out1",out1)
#        output = self.loss(derain_edge, gt_edge)
#        print("output",output/6/32/32)
#        print("output1",output)
#        output1 = self.loss1(derain_edge, gt_edge)
#        print("output2",output1)
        return out1
