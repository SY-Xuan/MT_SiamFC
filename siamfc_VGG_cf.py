from __future__ import absolute_import, division

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR

from got10k.trackers import Tracker

def Image_to_Tensor(img, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]):
    zn = np.asarray(img, 'float')
    zr = zn.transpose([2,0,1])
    for c in range(0, 3):
        zr[c] = ((zr[c]/255) - mean[c])/std[c]
    zt = torch.from_numpy(zr).float()
    return zt


def complex_mul(x, z):
    out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
    return torch.stack((out_real, out_imag), -1)


def complex_mulconj(x, z):
    out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_imag), -1)
def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(1, sz[0]+1) - np.floor(float(sz[0]) / 2), np.arange(1, sz[1]+1) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
    return g.astype(np.float32)


class VGG_Model(nn.Module):
    def __init__(self):
        super(VGG_Model, self).__init__()
        self.features1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.features2 = nn.Sequential(nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1))
        self.local = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)
    def forward(self, x):
        x1 = self.features1(x)
        out2 = self.features2(x1)
        out1 = self.local(x1)
        return out1, out2
class SiamFC(nn.Module):

    def __init__(self, is_train=True):
        super(SiamFC, self).__init__()
        self.features = VGG_Model()
        self.adjust = nn.BatchNorm2d(1)
        self._initialize_weights()
        self.is_train = is_train
        crop_sz = 239
        output_sz = 235
        self.lambda0 = 1e-4
        padding = 3.0
        output_sigma_factor = 0.1
        output_sigma = crop_sz / (1 + padding) * output_sigma_factor
        self.y = gaussian_shaped_labels(output_sigma, [output_sz, output_sz])
        self.yf = torch.rfft(torch.Tensor(self.y).view(1, 1, output_sz, output_sz).cuda(), signal_ndim=2)

    def forward(self, z, x):
        if self.is_train:
            z_tmp = z[:,:,56:183,56:183]
            _, z2 = self.features(z_tmp)
        z1, z = self.features(z) # 5
        x1, x = self.features(x) # 19

        
        
        # fast cross correlation
        n, c, h, w = x.size()
        x = x.view(1, n * c, h, w)
        out = F.conv2d(x, z2, groups=n)
        out = out.view(n, 1, out.size(-2), out.size(-1))

        # adjust the scale of responses
        #out = 0.001 * out + 0.0
        out = self.adjust(out)

        #correlation_filter
        if self.is_train:
            zf = torch.rfft(z1, signal_ndim=2)
            xf = torch.rfft(x1, signal_ndim=2)
            kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
            kxzf = torch.sum(complex_mulconj(xf, zf), dim=1, keepdim=True)
            alphaf = self.yf / (kzzf + self.lambda0)  # very Ugly
            response = torch.irfft(complex_mul(kxzf, alphaf), signal_ndim=2)
            return out, response
        else:
            return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode='fan_out',
                                     nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kargs):
        super(TrackerSiamFC, self).__init__(
            name='SiamFC', is_deterministic=True)
        self.cfg = self.parse_args(**kargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = SiamFC()
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)

        # setup lr scheduler
        self.lr_scheduler = ExponentialLR(
            self.optimizer, gamma=self.cfg.lr_decay)
        

    def parse_args(self, **kargs):
        # default parameters
        cfg = {
            # inference parameters
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            'adjust_scale': 0.001,
            # train parameters
            'initial_lr': 0.001, #change here 0.01->0.001
            'lr_decay': 0.8685113737513527,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}

        for key, val in kargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('GenericDict', cfg.keys())(**cfg)

    def init(self, image, box):
        self.net.is_train = False
        image = np.asarray(image)
        
        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(image, axis=(0, 1))
        exemplar_image = self._crop_and_resize(
            image, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            pad_color=self.avg_color)
           
        # exemplar features
        exemplar_image = Image_to_Tensor(exemplar_image).to(self.device).unsqueeze(0)
        #exemplar_image = torch.from_numpy(exemplar_image).to(
            #self.device).permute([2, 0, 1]).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            self.net.eval()
            _, self.kernel = self.net.features(exemplar_image)

    def update(self, image):
        self.net.is_train = False
        image = np.asarray(image)

        # search images
        instance_images = [self._crop_and_resize(
            image, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            pad_color=self.avg_color) for f in self.scale_factors]
        instance_images = [Image_to_Tensor(f).to(self.device).unsqueeze(0).squeeze(0) for f in instance_images]
        instance_images = torch.stack(instance_images)
        

        # responses
        with torch.set_grad_enabled(False):
            self.net.eval()
            _, instances = self.net.features(instance_images)
            responses = F.conv2d(instances, self.kernel) * 0.001
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            t, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC) for t in responses], axis=0)
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - self.upscale_sz // 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box

    def step(self, batch, backward=True, update_lr=False):
        self.net.is_train = True
        if backward:
            self.net.train()
            if update_lr:
                self.lr_scheduler.step()
        else:
            self.net.eval()

        z = batch[0].to(self.device)
        x = batch[1].to(self.device)
        label = batch[2].to(self.device)
        with torch.set_grad_enabled(backward):
            responses, out2 = self.net(z, x)
            labels, weights = self._create_labels(responses.size())
            loss1 = F.binary_cross_entropy_with_logits(
                responses, labels, weight=weights, size_average=True)
            loss2 = F.mse_loss(out2, label, size_average=True)
            
            loss = loss1 + loss2 * 5
            if backward:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = round(size)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)

        # crop image patch
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size))

        return patch

    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels, self.weights

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - w // 2
        y = np.arange(h) - h // 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # pos/neg weights
        pos_num = np.sum(labels == 1)
        neg_num = np.sum(labels == 0)
        weights = np.zeros_like(labels)
        weights[labels == 1] = 0.5 / pos_num
        weights[labels == 0] = 0.5 / neg_num
        weights *= pos_num + neg_num

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        weights = weights.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))
        weights = np.tile(weights, [n, c, 1, 1])

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        self.weights = torch.from_numpy(weights).to(self.device).float()

        return self.labels, self.weights
