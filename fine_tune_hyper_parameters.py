from __future__ import absolute_import, print_function

import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import models
from got10k.datasets import ImageNetVID, GOT10k
from siamfc_VGG_cf_add import TrackerSiamFC
from got10k_tmp.experiments import *
import numpy as np

tracker = TrackerSiamFC(net_path="./pretrained/siamfc_new/model_e1_BEST.pth")

experiments = [
        ExperimentOTB('/home/user/Documents/OTB100', version=2013),
        ExperimentOTB('/home/user/Documents/OTB100', version=2015), 
        ExperimentVOT('/home/user/Documents/vot2017/sequences', version=2017)
    ]
#fine tune the hyper-parameters
for i in np.arange(0.11, 0.12, 0.01):
    tracker.name += str(i)
    for e in experiments:
        tracker.cf_influence = i
        e.run(tracker, visualize=False)
        e.report([tracker.name])
