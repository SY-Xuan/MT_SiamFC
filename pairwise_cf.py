from __future__ import absolute_import, division

import numpy as np
from collections import namedtuple
from torch.utils.data import Dataset
from torchvision.transforms import Compose, CenterCrop, RandomCrop, ToTensor
from PIL import Image, ImageStat, ImageOps
import cv2
import random
import torch
def gaussian_shaped_labels(sigma, sz, shift):
    x, y = np.meshgrid(np.arange(1, sz[0]+1) - np.floor(float(sz[0]) / 2), np.arange(1, sz[1]+1) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1 + shift[0]), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1 + shift[1]), axis=1)
    return g.astype(np.float32)

def Image_to_Tensor(img, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]):
    zn = np.asarray(img, 'float')
    zr = zn.transpose([2,0,1])
    for c in range(0, 3):
        zr[c] = ((zr[c]/255) - mean[c])/std[c]
    zt = torch.from_numpy(zr).float()
    return zt






class RandomStretch(object):

    def __init__(self, max_stretch=0.05, interpolation='bilinear'):
        assert interpolation in ['bilinear', 'bicubic']
        self.max_stretch = max_stretch
        self.interpolation = interpolation

    def __call__(self, img):
        scale = 1.0 + np.random.uniform(
            -self.max_stretch, self.max_stretch)
        size = np.round(np.array(img.size, float) * scale).astype(int)
        if self.interpolation == 'bilinear':
            method = Image.BILINEAR
        elif self.interpolation == 'bicubic':
            method = Image.BICUBIC
        return img.resize(tuple(size), method)

class RandomCrop_change(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
       
        self.size = (int(size), int(size))
        
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, w - tw)
        j = random.randint(0, h - th)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        
        i, j, h, w = self.get_params(img, self.size)

        return img.crop((i, j, i + w, j + h)), i, j


class Pairwise(Dataset):

    def __init__(self, seq_dataset, **kargs):
        super(Pairwise, self).__init__()
        self.cfg = self.parse_args(**kargs)

        self.seq_dataset = seq_dataset
        self.indices = np.random.permutation(len(seq_dataset))
        # augmentation for exemplar and instance images
        self.transform_z = Compose([
            RandomStretch(max_stretch=0.05),
            CenterCrop(self.cfg.instance_sz - 8 * 2),
            
            
            ])
        self.transform_x = Compose([
            RandomStretch(max_stretch=0.05),
            CenterCrop(self.cfg.instance_sz - 8),
            ])
    def _shift_img(self, img, shift):
        #print(img.size())
        avg_color = ImageStat.Stat(img).mean
        search = np.array(img)
        # search = np.array(img).astype(np.float32)
        affine_arr = np.float32([[1,0,shift[0]],[0,1,shift[1]]])
        search = cv2.warpAffine(search, affine_arr, (search.shape[0],search.shape[1]), borderMode=cv2.BORDER_CONSTANT, borderValue=avg_color)
        
        return Image.fromarray(search)
    def parse_args(self, **kargs):
        # default parameters
        cfg = {
            'pairs_per_seq': 10,
            'max_dist': 100,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5}

        for key, val in kargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('GenericDict', cfg.keys())(**cfg)

    def __getitem__(self, index):
        index = self.indices[index % len(self.seq_dataset)]
        img_files, anno = self.seq_dataset[index]
        rand_z, rand_x = self._sample_pair(len(img_files))
        to_tensor = ToTensor()
        exemplar_image = Image.open(img_files[rand_z])
        instance_image = Image.open(img_files[rand_x])
        exemplar_image = self._crop_and_resize(exemplar_image, anno[rand_z])
        instance_image = self._crop_and_resize(instance_image, anno[rand_x])

        
        exemplar_image = self.transform_z(exemplar_image)
        instance_image = self.transform_x(instance_image)

        random_crop_x = RandomCrop_change(self.cfg.instance_sz - 16)
        instance_image, xx, xy = random_crop_x(instance_image)
        center_x_shift = ((self.cfg.instance_sz - 9) / 2 - (self.cfg.instance_sz - 16 - 1) / 2 - xx, (self.cfg.instance_sz - 9) / 2 - (self.cfg.instance_sz - 17) / 2 - xy)
        instance_image = self._shift_img(instance_image, -np.array(center_x_shift))
        
        instance_image = Image_to_Tensor(instance_image)
        exemplar_image = Image_to_Tensor(exemplar_image)
        # 5.8 is pre-computed based on the KCF
        label = gaussian_shaped_labels(5.8, (235, 235), (0, 0))
        return exemplar_image, instance_image, np.array([label])

    def __len__(self):
        return self.cfg.pairs_per_seq * len(self.seq_dataset)

    def _sample_pair(self, n):
        assert n > 0
        if n == 1:
            return 0, 0
        elif n == 2:
            return 0, 1
        else:
            max_dist = min(n - 1, self.cfg.max_dist)
            rand_dist = np.random.choice(max_dist) + 1
            rand_z = np.random.choice(n - rand_dist)
            rand_x = rand_z + rand_dist

        return rand_z, rand_x

    def _crop_and_resize(self, image, box):
        # convert box to 0-indexed and center based
        box = np.array([
            box[0] - 1 + (box[2] - 1) / 2,
            box[1] - 1 + (box[3] - 1) / 2,
            box[2], box[3]], dtype=np.float32)
        center, target_sz = box[:2], box[2:]

        # exemplar and search sizes
        context = self.cfg.context * np.sum(target_sz)
        z_sz = np.sqrt(np.prod(target_sz + context))
        x_sz = z_sz * self.cfg.instance_sz / self.cfg.exemplar_sz

        # convert box to corners (0-indexed)
        size = round(x_sz)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.size))
        npad = max(0, int(pads.max()))
        if npad > 0:
            avg_color = ImageStat.Stat(image).mean
            # PIL doesn't support float RGB image
            avg_color = tuple(int(round(c)) for c in avg_color)
            image = ImageOps.expand(image, border=npad, fill=avg_color)

        # crop image patch
        corners = tuple((corners + npad).astype(int))
        patch = image.crop(corners)

        # resize to instance_sz
        out_size = (self.cfg.instance_sz, self.cfg.instance_sz)
        patch = patch.resize(out_size, Image.BILINEAR)

        return patch
