import torch
import time
import os
from torchvision.models import resnet18, mobilenet_v2
from loguru import logger

from zraysched import ApplicationActor
from examples.two_classification_apps.data import get_caltech256_dataloader, get_domainnet_dataloader


class DemoApplication_ResNet18(ApplicationActor):
    def init_model(self):
        return resnet18(pretrained=True)
    def get_dataloader_func(self):
        dataloaders_func = [get_caltech256_dataloader, get_domainnet_dataloader]
        return dataloaders_func[self.distribution_index % len(dataloaders_func)]
    

class DemoApplication_MobileNet(ApplicationActor):
    def init_model(self):
        return mobilenet_v2(pretrained=True)
    def get_dataloader_func(self):
        dataloaders_func = [get_domainnet_dataloader, get_caltech256_dataloader]
        return dataloaders_func[self.distribution_index % len(dataloaders_func)]
    