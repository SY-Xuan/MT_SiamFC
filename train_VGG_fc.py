from __future__ import absolute_import, print_function

import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import models
from got10k.datasets import ImageNetVID, GOT10k
from pairwise_cf import Pairwise
from siamfc_VGG_cf import TrackerSiamFC
from got10k_tmp.experiments import *

if __name__ == '__main__':
    # setup dataset
    name = 'VID'
    assert name in ['VID', 'GOT-10k']
    if name == 'GOT-10k':
        root_dir = 'data/GOT-10k'
        seq_dataset = GOT10k(root_dir, subset='train')
    elif name == 'VID':
        root_dir = '/home/user/ILSVRC2015'
        seq_dataset = ImageNetVID(root_dir, subset=('train', 'val'))
    pair_dataset = Pairwise(seq_dataset)

    # setup data loader
    cuda = torch.cuda.is_available()
    loader = DataLoader(
        pair_dataset, batch_size=8, shuffle=True,
        pin_memory=cuda, drop_last=True, num_workers=4)

    # setup tracker
    tracker = TrackerSiamFC()

    #pretrained vgg
    model_vgg = models.vgg16(pretrained=True)
    model_state = tracker.net.state_dict()
    vgg_dict = model_vgg.state_dict()
    
    pretrained_dict = {'features.features1.0.bias':vgg_dict['features.0.bias'], 'features.features1.0.weight':vgg_dict['features.0.weight'], 'features.features1.2.weight':vgg_dict['features.2.weight'], 'features.features1.2.bias':vgg_dict['features.2.bias'],  'features.features2.2.bias':vgg_dict['features.5.bias'], 'features.features2.2.weight':vgg_dict['features.5.weight'], 'features.features2.4.bias':vgg_dict['features.7.bias'], 'features.features2.4.weight':vgg_dict['features.7.weight'], 'features.features2.7.bias':vgg_dict['features.10.bias'], 'features.features2.7.weight':vgg_dict['features.10.weight'], 'features.features2.9.bias':vgg_dict['features.12.bias'], 'features.features2.9.weight':vgg_dict['features.12.weight'], 'features.features2.11.weight':vgg_dict['features.14.weight'], 'features.features2.11.bias':vgg_dict['features.14.bias'], 'features.features2.14.weight':vgg_dict['features.17.weight'], 'features.features2.14.bias':vgg_dict['features.17.bias'], 'features.features2.16.weight':vgg_dict['features.19.weight'], 'features.features2.16.bias':vgg_dict['features.19.bias'], 'features.features2.18.weight':vgg_dict['features.21.weight'], 'features.features2.18.bias':vgg_dict['features.21.bias']}
    model_state.update(pretrained_dict)
    tracker.net.load_state_dict(model_state)


    # path for saving checkpoints
    net_dir = 'pretrained/siamfc_new'
    if not os.path.exists(net_dir):
        os.makedirs(net_dir)
    best_auc = 0
    # training loop
    epoch_num = 50
    for epoch in range(epoch_num):
        for step, batch in enumerate(loader):
            loss = tracker.step(
                batch, backward=True, update_lr=(step == 0))
            if step % 100 == 0:
                print('Epoch [{}][{}/{}]: Loss: {:.3f}'.format(
                    epoch + 1, step + 1, len(loader), loss))
                

        # save checkpoint
        net_path = os.path.join(net_dir, 'model_e%d.pth' % (epoch + 1))
        
        torch.save(tracker.net.state_dict(), net_path)
        test_tracker = TrackerSiamFC(net_path=net_path)
        e = ExperimentOTB("/home/user/OTB100", version=2015)
        e.run(test_tracker, visualize=False)
        auc = e.report([test_tracker.name])
        if auc > best_auc:
            net_path2 = os.path.join(net_dir, 'model_e%d_BEST.pth' % (epoch + 1))
            torch.save(tracker.net.state_dict(), net_path2)
            best_auc = auc
        print("now_best:{}".format(best_auc))
