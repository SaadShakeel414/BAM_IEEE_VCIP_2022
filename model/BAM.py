# -*- coding: utf-8 -*-
"""
Created on Tue May  3 23:25:10 2022

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from math import sqrt

    
################################Bidirectional Attention Module###########################

class BAM(nn.Module):
    def __init__(self,in_channels,r = 4):
        super(BAM,self).__init__()
        
        self.conva = nn.Conv2d(in_channels,in_channels // r, 1)   ###[B,128,7,7]
        self.convb = nn.Conv2d(in_channels,in_channels // r, 1)   ###[B,128,7,7]
        self.relu = nn.ReLU(inplace = True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.CA = nn.Sequential(nn.Conv2d(in_channels,in_channels // 4, kernel_size = 1, padding =0, bias = True),
                                nn.ReLU(inplace = True),
                                nn.Conv2d(in_channels // 4, in_channels, kernel_size = 1, padding =0, bias = True)
                                ,nn.Sigmoid())
        self.dense = nn.Conv2d(128,512,kernel_size=1,padding=0,bias=False)
        
        
    def forward(self,x):
        
        b,c,w,h = x.size()
        feaX1 = x.permute(0,2,1,3).contiguous()
        feaX1 = feaX1.view(b,-1,w,h)
        feaX1 = self.relu(self.conva(feaX1))
        feaY1 = self.relu(self.convb(x))
        
        d1 = feaX1.size(-1)
        d2 = feaY1.size(-1)

        feaXY1 = torch.matmul(feaX1,feaY1.transpose(-2,-1)) / sqrt(d1)  ########
        AttfeaXY1 = F.softmax(feaXY1,dim = -1)
        
        AttfeaXY11 = torch.matmul(feaY1, AttfeaXY1)  #######In the next code, i will multiply with feaY (as its transpose is being taken)

        feaYX1 = torch.matmul(feaY1,feaX1.transpose(-2,-1)) / sqrt(d2)
        AttfeaYX11 = F.softmax(feaYX1,dim = -1)
        
        AttfeaYX11 = torch.matmul(feaX1, AttfeaYX11)  #######3In the next code, i will multiply with feaX (as its transpose is being taken)
        
        Att_C1 = AttfeaXY11 + AttfeaYX11
        Att_C1 = self.dense(Att_C1)
############################Second Branch###########################################
        
        feaX2 = x.permute(0,3,2,1).contiguous()
        feaX2 = feaX2.view(b,-1,w,h)
        feaX2 = self.relu(self.conva(feaX2))
        
        feaY2 = self.relu(self.convb(x))
        
        d1 = feaX2.size(-1)
        d2 = feaY2.size(-1)
        
        feaXY2 = torch.matmul(feaX2,feaY2.transpose(-2,-1)) / sqrt(d1)  ########
        AttfeaXY2 = F.softmax(feaXY2,dim = -1)
        
        AttfeaXY22 = torch.matmul(feaY2, AttfeaXY2)
        
        feaYX2 = torch.matmul(feaY2,feaX2.transpose(-2,-1)) / sqrt(d2)
        AttfeaYX22 = F.softmax(feaYX2,dim = -1)
        
        AttfeaYX22 = torch.matmul(feaX2, AttfeaYX22)
        
        Att_C2 = AttfeaXY22 + AttfeaYX22
        Att_C2 = self.dense(Att_C2)

        AP1 = self.avg_pool(Att_C1)
        Cha_Des1 = self.CA(AP1)
        
        AP2 = self.avg_pool(Att_C2)
        Cha_Des2 = self.CA(AP2)

        return (Cha_Des1 + Cha_Des2) + x
        
        
class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    def forward(self, inputs):
        return self.op(inputs)             

# ---------------------------------- LResNet50E-IR network Begin ----------------------------------

class BlockIR(nn.Module):
#    pooling_r = 1
    def __init__(self, inplanes, planes, stride, dim_match):
        super(BlockIR, self).__init__()
#        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)       
        self.bn3 = nn.BatchNorm2d(planes)
        
        if dim_match:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class LResNet_BAM_MFR(nn.Module):

    def __init__(self, block, layers, filter_list, is_gray=False, cardinality = 1):
        self.inplanes = 64
        super(LResNet_BAM_MFR, self).__init__()
        # input is (mini-batch,3 or 1,112,96)
        # use (conv3x3, stride=1, padding=1) instead of (conv7x7, stride=2, padding=3)
        if is_gray:
            self.conv1 = nn.Conv2d(1, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)  # gray
        else:
            self.conv1 = nn.Conv2d(3, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        self.prelu1 = nn.PReLU(filter_list[0])
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.sigmoid = nn.Sigmoid()
        self.projector = ProjectorBlock(128, 512)
        

#        self.Att = DAM(512,4)
        self.BAM_FR = BAM(512)
        
        self.fc = nn.Sequential(
            nn.BatchNorm1d(512 * 7 * 7),
            nn.Dropout(p=0.4),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512),  # fix gamma ???
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)


    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(block(inplanes, planes, stride, False))
        for i in range(1, blocks):
            layers.append(block(planes, planes, stride=1, dim_match=True))

        return nn.Sequential(*layers)
    
    

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)

        x = self.layer1(x)
        x = self.layer2(x) 
        
        fea_int = x

        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.BAM_FR(x)
        
        feature2 = self.projector(fea_int)
        feature2 = self.avgpool(feature2) 
        
        fea_Att = x
        
        fus_fea = fea_Att + feature2
        wei = self.sigmoid(fus_fea)
        x = wei * fus_fea

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


def LResNet50E_IR_BAM(is_gray=False):
    filter_list = [64, 64, 128, 256, 512]
    layers = [3, 4, 14, 3]
    return LResNet_BAM_MFR(BlockIR, layers, filter_list, is_gray)
# ---------------------------------- LResNet50E-IR network End ----------------------------------
