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

from configs import CambridgeConfig

import torch


class CambridgeSegmentation(Dataset):
    """
    Cambridge dataset
    """
    seg_channel = 3
    VERTEX_CHANNEL = 2

    def __init__(self,
                 cfg,
                 split="train",
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
        _image_suffix = ".png"
        _seg_suffix = ".seg.png"
        _vertex_suffix = ".vertex_2d.npy"

        self._extrinsics = json.load(
            open(osp.join(self._data_dir, 'out_extrinsics.json')))

        # read id2centers.json
        id2centers = json.load(
            open(osp.join(self._data_dir, "id2centers.json")))
        self._id2centers = np.array(id2centers)

        self._camera_matrix = np.array(
            [[0, 0, 960], [0, 0, 540], [0, 0, 1.0]]).astype(np.float64)

        self.image_list = []
        self.seg_target_list = []
        self.vertex_target_list = []

        lines = []
        # load train/val lines
        if split == 'train':
            split_fn = osp.join(self._data_dir, 'train_list.json')
        else:
            split_fn = osp.join(self._data_dir, 'test_list.json')

        lines.extend(json.load(open(split_fn)))

        # lines = train_list

        # accelerate debug
        if cfg["validation_debug"]:
            lines = lines[:20]

        for ii, line in enumerate(lines):
            _image = osp.join(self._data_dir, line)
            _seg_target = osp.join(
                self._data_dir, line.replace(_image_suffix, _seg_suffix))
            _vertex_target = osp.join(
                self._data_dir, line.replace(_image_suffix, _vertex_suffix))

            assert osp.isfile(_image)
            # assert osp.isfile(_seg_target)
            # assert osp.isfile(_vertex_target)

            self.image_list.append(_image)
            self.seg_target_list.append(_seg_target)
            self.vertex_target_list.append(_vertex_target)

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
            # pixel (1920, 1080)
            _ori_img = cv2.imread(self.image_list[index])
            if osp.exists(self.seg_target_list[index]):
                _seg_target = cv2.imread(
                    self.seg_target_list[index])        # pixel (960, 540)
            else:
                _seg_target = np.zeros_like(
                    _ori_img[::2, ::2, :])            # pixel (960, 540)
            # print(self.seg_target_list[index])
            # _vertex_target = np.load(self.vertex_target_list[index])
            if osp.exists(self.vertex_target_list[index]):
                # (480, 270) or # (960, 540)
                _vertex_target = np.load(self.vertex_target_list[index])
            else:
                _vertex_target = np.zeros_like(_seg_target)[..., :2]
            file_id = self.image_list[index].split(self.cfg["scene"] + '/')[1]
            _pose_target = np.diag((1.0,) * 4).astype(np.float64)
            if file_id in self._extrinsics.keys():
                _pose_target[:3, :] = np.array(self._extrinsics[file_id][:12]).astype(
                    np.float64).reshape(3, 4)  # world to camera

            _img = _ori_img
            # pixel (1920, 1080) --> (960, 540)
            _img = _img[::2, ::2, :]
            _ori_img = _ori_img[::2, ::2, :]

            # bgr to rgb
            _ori_img = cv2.cvtColor(_ori_img, cv2.COLOR_BGR2RGB)
            _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)

            # segmenataion label transfer: rgb --> int
            _seg_target = cv2.cvtColor(
                _seg_target, cv2.COLOR_BGR2RGB).astype(np.int32)
            _seg_target = _seg_target[:, :, 0] + _seg_target[:, :, 1] * 256
            _seg_target[_seg_target >= len(self._id2centers)] = 0

            # camera intrinsics
            # _camera_k_matrix = self.intrinsics[file_id]
            _camera_k_matrix = self._camera_matrix.copy()
            _camera_k_matrix[0, 0] = _camera_k_matrix[1,
                                                      1] = self._extrinsics[file_id][12]
            # _camera_k_matrix[0, 0] = _camera_k_matrix[1, 1] = 1671.31042480469
            scale = 2
            _camera_k_matrix[:2, :] = _camera_k_matrix[:2, :] / scale

            if self.split == "train":
                r_vec, _ = cv2.Rodrigues(_pose_target[:3, :3])
                t_vec = _pose_target[:3, 3:]

                _mask = np.ones(_seg_target.shape, np.float32)
                _img, _seg_target, _vertex_target, _mask = tr.affine_aug(
                    _img, _seg_target, self._id2centers, r_vec, t_vec, _camera_k_matrix, self.cfg["use_aug"])
            else:
                _mask = None

        except Exception as e:
            print(self.image_list[index])
            print(self.image_list[index])
            print(self.seg_target_list[index])
            print('vertex:', self.vertex_target_list[index])
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
        return "cambridge(split=" + str(self.split) + ")"
