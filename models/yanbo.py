import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vgg as vgg
import models.resnet as resnet
import models.vision_transformer as vit
from models.inception import inception_v3, BasicConv2d

import timm
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
# from torch.nn.modules.pooling import AdaptiveAvgPool2d
import ml_collections


class YANBO(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        self.num_classes = num_classes

        if 'resnet' in net:
            self.features = getattr(resnet, net)(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        else:
            raise ValueError('Unsupported net:%s' % net)
        
        self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)

    def forward(self, x):
        batch_size = x.size(0)

        feature_maps = self.features(x)
        p = self.fc(feature_maps)
        return p

class Classifier(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.encoder = timm.create_model(model_name=self.backbone, in_chans=3, pretrained=self.pretrained)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.dropout = Dropout(self.dropout_rate)
        self.fc = Linear(self.encoder.num_features, self.num_classes)
    
    def forward(self, x):
        x = self.encoder.forward_features(x)
        if 'vit' in self.backbone:
            pass
        else:
            x = self.avg_pool(x).flatten(1)
        if self.training:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class ClassifierViT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        config = vit.get_config(self.backbone)
        self.encoder = vit.VisionTransformer(config, self.image_size, num_classes=self.num_classes)

        if self.pretrained:
            self.encoder.load_from(np.load("./ViT-B_16.npz"))

        self.dropout = Dropout(self.dropout_rate)
        self.fc = Linear(self.encoder.num_features, self.num_classes)
    
    def forward(self, x):
        x = self.encoder.forward_features(x)
        if self.training:
            x = self.dropout(x)
        x = self.fc(x)
        return x
