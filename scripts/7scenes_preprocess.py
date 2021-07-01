#!/usr/bin/env python
# coding: utf-8

from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from matplotlib.colors import LinearSegmentedColormap
import torch
from tqdm import tqdm
from glob import glob
import numpy as np
import os.path as osp
import json
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description="result statistics")
parser.add_argument("--scene", type=str, default="chess", help="scene")
parser.add_argument("--size", type=str, default="480*640",
                    choices=["480*640", "240*320"],
                    help="generate image size")
parser.add_argument("--root_dir", type=str, default="/home/zhouhan/data/7scenes_release", help="data root dir")
parser.add_argument("--ver_png", type=str, default="false",
                    choices=["true", "false"], help="save vertex png")
args = parser.parse_args()

work_dir = args.root_dir


cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.2, 0.2, 0.2),
                 (0.4, 0.0, 0.0),
                 (0.6, 1.0, 1.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                   (0.2, 1.0, 1.0),
                   (0.4, 1.0, 1.0),
                   (0.6, 1.0, 1.0),
                   (0.8, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.2, 1.0, 1.0),
                  (0.4, 0.0, 0.0),
                  (0.6, 0.0, 0.0),
                  (0.8, 0.0, 0.0),
                  (1.0, 1.0, 1.0))}

cmap = LinearSegmentedColormap('Rd_Bl_Rd', cdict, 350)


k_matrix = np.array([[520, 0, 320],
                    [0, 520, 240],
                    [0, 0, 1]]).astype(np.float64)
scale = 1
if args.size != "480*640":
    scale = 2
k_matrix[:2, :] = k_matrix[:2, :] / scale
print(k_matrix)

scenes = os.listdir(work_dir)
for scene in scenes:
    if not osp.isdir(osp.join(work_dir, scene)):
        continue
    if scene == 'segmentations':
        continue
    if args.scene != "" and scene != args.scene:
        continue
    print('process project %s !' % scene)
    id2center = json.load(
        open(osp.join(work_dir, scene, 'id2centers.json')))
    id2center = np.array(id2center).astype(np.float64)
    for file in tqdm(sorted(glob(osp.join(work_dir, scene, 'seq-*/frame*.seg.png')))):
        label = cv2.imread(file)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB).astype(np.int64)
        label = label[:, :, 0] * 256 * 256 + \
            label[:, :, 1] * 256 + label[:, :, 2]
        label[label >= len(id2center)] = 0
        if args.size != "480*640":
            label = label[::2, ::2]

        pose_file = file.replace('seg.png', 'pose.txt')
        pose = np.loadtxt(pose_file).astype(np.float64)
        pose[0, 3] = pose[0, 3] + 0.0245
        # pose[:3, 3] = pose[:3, 3] + np.matmul(pose[:3, :3], np.matmul(pose[:3, :3], np.array([0.0245, 0, 0])))
        pose = np.linalg.inv(pose)
        r_vec, _ = cv2.Rodrigues(pose[:3, :3])
        t_vec = pose[:3, 3:]

        height, width = label.shape
        # pt2d, _ = cv2.projectPoints(id2center[label].reshape(-1, 3), r_vec, t_vec, np.array(k_matrix), None)
        pt2d, _ = cv2.projectPoints(
            id2center.reshape(-1, 3), r_vec, t_vec, np.array(k_matrix), None)
        pt2d = pt2d[label]
        pt2d = pt2d.reshape(height, width, 2)

        vertex_2d = np.zeros((height, width, 2), dtype=np.float64)
        vertex_2d[..., 0] = np.tile(
            np.arange(width, dtype=np.float64), height).reshape(height, width)
        vertex_2d[..., 1] = np.tile(np.arange(height, dtype=np.float64), width).reshape(
            width, height).transpose(1, 0)

        vertex_2d = pt2d - vertex_2d

        length = np.sqrt(vertex_2d[..., 0] ** 2 + vertex_2d[..., 1] ** 2)
        vertex_2d[..., 0] = vertex_2d[..., 0] / length
        vertex_2d[..., 1] = vertex_2d[..., 1] / length
        vertex_2d[label == 0] = 0
        vertex2d_file = file.replace('seg.png', 'vertex_2d.npy')
        np.save(vertex2d_file, vertex_2d)

        if args.ver_png == "true":
            vertex_1d = np.arctan2(vertex_2d[..., 1], vertex_2d[..., 0])
            vertex_1d = (vertex_1d * 180/np.pi).astype(np.int) + 180

            vertex_1d = (cmap(vertex_1d)[..., :3] * 255).astype(np.uint8)
            vertex_1d[label == 0] = 0
            cv2.imwrite(file.replace('seg.png', 'vertex.png'), vertex_1d)
