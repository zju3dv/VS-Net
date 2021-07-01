from __future__ import print_function, division
import os
import cv2
import json
import random
import numpy as np
import os.path as osp
from glob import glob
from PIL import Image
from matplotlib import cm
from torch.utils.data import Dataset
from dataloaders import custom_transforms as tr

from configs import SevenScenesConfig

import torch


class SevenScenesSegmentation(Dataset):
    """
    SevenScenes dataset
    """
    seg_channel = 3
    VERTEX_CHANNEL = 2

    def __init__(self,
                 cfg,
                 split="training",
                 ):
        """
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self.cfg = cfg
        self.split = split
        self._data_dir = cfg["data_dir"]
        self.seg_channel = cfg["seg_channel"]
        self.VERTEX_CHANNEL = cfg["vertex_channel"]
        _image_suffix = ".color.png"
        _seg_suffix = ".seg.png"
        _vertex_suffix = ".vertex_2d.npy"
        _pose_suffix = ".pose.txt"

        # read id2centers.json
        id2centers = json.load(
            open(osp.join(self._data_dir, "id2centers.json")))
        self._id2centers = np.array(id2centers)

        self.image_list = []
        self.seg_target_list = []
        self.vertex_target_list = []
        self.pose_target_list = []

        # load train/val sequences
        if split == 'train':
            split_fn = osp.join(self._data_dir, 'TrainSplit.txt')
        else:
            split_fn = osp.join(self._data_dir, 'TestSplit.txt')
        sequences = []
        with open(split_fn) as f:
            sequences.extend([line.replace('\n', '')
                             for line in f.readlines()])

        # prepare train/val file list
        lines = []
        for seq in sequences:
            lines.extend(
                sorted(glob(osp.join(self._data_dir, seq, '*.color.png'))))
        lines = [f.replace('.color.png', '') for f in lines]

        # accelerate debug
        if cfg["validation_debug"]:
            lines = lines[:20]

        for ii, line in enumerate(lines):
            _image = line + _image_suffix
            _seg_target = line + _seg_suffix
            _vertex_target = line + _vertex_suffix
            _pose_target = line + _pose_suffix

            assert osp.isfile(_image)
            assert osp.isfile(_seg_target)
            # assert osp.isfile(_vertex_target)
            assert osp.isfile(_pose_target)

            self.image_list.append(_image)
            self.seg_target_list.append(_seg_target)
            self.vertex_target_list.append(_vertex_target)
            self.pose_target_list.append(_pose_target)

        # adjust camera intrisics
        self._camera_k_matrix = np.array([[520, 0, 320],
                                          [0, 520, 240],
                                          [0, 0, 1]]).astype(np.float64)

        # Display stats
        print("Number of images in {}: {:d}".format(
            split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # load data
        sample = self._make_img_gt_point_pair(index)

        # data transform
        if self.split == "train":
            sample = self.transform_tr(sample)
        else:
            sample = self.transform_val(sample)[:-1]

        return sample

    def _make_img_gt_point_pair(self, index):
        try:
            # opencv read bgr
            # (480, 640)
            _ori_img = cv2.imread(self.image_list[index])
            _seg_target = cv2.imread(
                self.seg_target_list[index])           # (480, 640)
            if osp.exists(self.vertex_target_list[index]):
                # (480, 640) or # (240, 320)
                _vertex_target = np.load(self.vertex_target_list[index])
            else:
                _vertex_target = np.zeros_like(_seg_target)[..., :2]
            _pose_target = np.loadtxt(self.pose_target_list[index]).astype(
                np.float64)  # word to camera

            _img = _ori_img

            # bgr to rgb
            _ori_img = cv2.cvtColor(_ori_img, cv2.COLOR_BGR2RGB)
            _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)

            # segmenataion label transfer: rgb --> int
            _seg_target = cv2.cvtColor(
                _seg_target, cv2.COLOR_BGR2RGB).astype(np.int32)
            _seg_target = _seg_target[:, :, 0] * 256 * 256 + \
                _seg_target[:, :, 1] * 256 + _seg_target[:, :, 2]
            _seg_target[_seg_target >= len(self._id2centers)] = 0

            # adjust camera extrinsics
            _pose_target[0, 3] = _pose_target[0, 3] + 0.0245
            _pose_target = np.linalg.inv(_pose_target)
            # camera intrinsics
            _camera_k_matrix = self._camera_k_matrix

            if self.split == "train":
                r_vec, _ = cv2.Rodrigues(_pose_target[:3, :3])
                t_vec = _pose_target[:3, 3:]

                _img, _seg_target, _vertex_target, _mask = tr.affine_aug(
                    _img, _seg_target, self._id2centers, r_vec, t_vec, _camera_k_matrix, self.cfg["use_aug"])
            else:
                _mask = None

        except Exception as e:
            print(self.image_list[index])
            print(self.seg_target_list[index])
            print(self.vertex_target_list[index])
            print(self.pose_target_list[index])
            raise e

        return _img, _seg_target, _vertex_target, _pose_target, _camera_k_matrix, _ori_img, _mask

    def transform_tr(self, sample):
        augmentation = []
        # use data usual transform
        augmentation.append(tr.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        augmentation.append(tr.ToTensor())
        composed_transforms = tr.Compose(augmentation)

        return composed_transforms(*sample)

    def transform_val(self, sample):
        augmentation = []
        # use data usual transform
        augmentation.append(tr.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        augmentation.append(tr.ToTensor())
        composed_transforms = tr.Compose(augmentation)

        return composed_transforms(*sample)

    def __str__(self):
        return "sevenscenes(split=" + str(self.split) + ")"
