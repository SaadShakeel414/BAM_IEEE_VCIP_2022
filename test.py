# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 22:50:42 2021

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.nn.functional as F
from config import configurations
from model.BAM import LResNet50E_IR_BAM

import sys
sys.path.append("..")
from loss.focal_loss import FocalLoss
from util.utils import get_val_data, perform_val, get_time, AverageMeter, accuracy
from tqdm import tqdm
import os
#import time
#import numpy as np
#import scipy
#import pickle
import config as cfg
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    #======= hyperparameters & data loaders =======#
    cfg = configurations[1]
    torch.backends.cudnn.benchmark = True
    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)

    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    HEAD_NAME = cfg['HEAD_NAME'] # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    INPUT_SIZE = cfg['INPUT_SIZE']
    BATCH_SIZE = cfg['BATCH_SIZE']
    RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    GPU_ID = cfg['GPU_ID'] # specify your GPU ids
    print("Overall Configurations:")
    print(cfg)
    val_data_dir = os.path.join(DATA_ROOT, 'val_data')
    
    cplfw, cplfw_issame, calfw, calfw_issame, cfp_fp, cfp_fp_issame, lfw, lfw_issame, agedb_30, agedb_30_issame, cfp_ff, cfp_ff_issame, vgg2_fp, vgg2_fp_issame = get_val_data(val_data_dir)

    model = LResNet50E_IR_BAM(False)
    model_path_BAM = ''    ########Enter the path for checkpoint file (.pth) here##########
    model.load_state_dict(torch.load(model_path_BAM))
    model.cuda()


    accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(False, 0, EMBEDDING_SIZE, BATCH_SIZE, model, calfw, calfw_issame, nrof_folds = 10, tta = True)
    accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(False, 0, EMBEDDING_SIZE, BATCH_SIZE, model, cplfw, cplfw_issame, nrof_folds = 10, tta = True)
    accuracy_agedb30, best_threshold_agedb30, roc_curve_agedb30 = perform_val(False, 0, EMBEDDING_SIZE, BATCH_SIZE, model, agedb_30, agedb_30_issame, nrof_folds = 10, tta = True)
    accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(False, 0, EMBEDDING_SIZE, BATCH_SIZE, model, lfw, lfw_issame, nrof_folds = 10, tta = True)
    accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(False, 0, EMBEDDING_SIZE, BATCH_SIZE, model, cfp_fp, cfp_fp_issame, nrof_folds = 10, tta = True)
    accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(False, 0, EMBEDDING_SIZE, BATCH_SIZE, model, cfp_ff, cfp_ff_issame, nrof_folds = 10, tta = True)
    accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(False, 0, EMBEDDING_SIZE, BATCH_SIZE, model, vgg2_fp, vgg2_fp_issame, nrof_folds = 10, tta = True)
    accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(EMBEDDING_SIZE, BATCH_SIZE, model, cplfw, cplfw_issame)


    print("LFW Acc: {}".format(accuracy_lfw))
    print("CFP-FF Acc: {}".format(accuracy_cfp_ff))
    print("CFP-FP Acc: {}".format(accuracy_cfp_fp))
    print("AgeDB30 Acc: {}".format(accuracy_agedb30))
    print("CPLFW Acc: {}".format(accuracy_cplfw))
    print("CALFW Acc: {}".format(accuracy_calfw))
    print("VGGFace2_FP Acc: {}".format(accuracy_vgg2_fp))
    #
    
    

