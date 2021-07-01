#!/usr/bin/env python
# coding: utf-8

from sklearn.manifold import Isomap
from sklearn.manifold import MDS
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import torch
import json
import numpy as np
from tqdm import tqdm
from glob import glob
import os.path as osp
import cv2
import argparse

parser = argparse.ArgumentParser(description="result statistics")
parser.add_argument("--scene", type=str, default="GreatCourt", help="scene")
parser.add_argument("--size", type=str, default="960*540",
                    choices=["1920*1080", "960*540", "480*270"],
                    help="generate image size")
parser.add_argument("--root_dir", type=str,
                    default="/home/zhouhan/data/cambridge_release", help="data root dir")
parser.add_argument("--ver_png", type=str, default="false",
                    choices=["true", "false"], help="save vertex png")
args = parser.parse_args()


cdict = {"red": ((0.0, 0.0, 0.0),
                 (0.2, 0.2, 0.2),
                 (0.4, 0.0, 0.0),
                 (0.6, 1.0, 1.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
         "green": ((0.0, 0.0, 0.0),
                   (0.2, 1.0, 1.0),
                   (0.4, 1.0, 1.0),
                   (0.6, 1.0, 1.0),
                   (0.8, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         "blue": ((0.0, 1.0, 1.0),
                  (0.2, 1.0, 1.0),
                  (0.4, 0.0, 0.0),
                  (0.6, 0.0, 0.0),
                  (0.8, 0.0, 0.0),
                  (1.0, 1.0, 1.0))}
vertex_cmap = LinearSegmentedColormap("Rd_Bl_Rd", cdict, 350)

work_dir = args.root_dir
scenes = os.listdir(work_dir)

camera_matrix = np.array(
    [[0, 0, 960], [0, 0, 540], [0, 0, 1.0]]).astype(np.float64)
for scene in scenes:
    if not osp.isdir(osp.join(work_dir, scene)):
        continue
    if args.scene != "" and scene != args.scene:
        continue
    print('process project %s !' % scene)

    id2center = json.load(
        open(osp.join(work_dir, scene, 'id2centers.json')))
    id2center = np.array(id2center).astype(np.float64)
    extrinsics = json.load(
        open(osp.join(work_dir, scene, 'out_extrinsics.json')))
    unique_label_list = np.array([])
    for i, file in enumerate(tqdm(sorted(glob(osp.join(work_dir, scene, '*/*.seg.png'))))):
        label = cv2.imread(file)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB).astype(np.int64)
        label = label[:, :, 0] + label[:, :, 1] * 256
        label[label >= len(id2center)] = 0
        if args.size == "480*270":
            label = label[::2, ::2]

        unique_label = np.unique(label)
        file_id = file.split(scene + '/')[1].replace('seg.', '')
        pose = np.diag((1.0,) * 4).astype(np.float64)
        pose[:3, :] = np.array(extrinsics[file_id][:12]).astype(
            np.float64).reshape(3, 4)

        r_vec, _ = cv2.Rodrigues(pose[:3, :3])
        t_vec = pose[:3, 3:]

        scale = 2
        k_matrix = camera_matrix.copy()
        k_matrix[0, 0] = k_matrix[1, 1] = extrinsics[file_id][12]
        if args.size == "480*270":
            scale = 4
        k_matrix[:2, :] = k_matrix[:2, :] / scale

        global_pt2d = np.zeros((id2center.shape[0], 2), dtype=np.float64)

        height, width = label.shape
        unique_label = np.unique(label)
        unique_pt2d, _ = cv2.projectPoints(
            id2center[unique_label], r_vec, t_vec, k_matrix, None)

        global_pt2d[unique_label] = np.squeeze(unique_pt2d)
        pt2d = global_pt2d[label]

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
        np.save(file.replace('seg.png', 'vertex_2d.npy'), vertex_2d)

        if args.ver_png == "true":
            vertex_1d = np.arctan2(vertex_2d[..., 1], vertex_2d[..., 0])
            vertex_1d = (vertex_1d * 180/np.pi).astype(np.int) + 180

            vertex_1d = (vertex_cmap(vertex_1d)
                            [..., :3] * 255).astype(np.uint8)
            vertex_1d[label == 0] = 0
            cv2.imwrite(file.replace('seg.png', 'vertex.png'), vertex_1d)
