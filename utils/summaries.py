import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import cv2

import numpy as np

from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

from utils import metrics

cdict = {"red": ((0.0, 0.0, 0.0),
                 (0.2, 0.2, 0.2),
                 (0.4, 0.0, 0.0),
                 (0.6, 1.0, 1.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
        "green":((0.0, 0.0, 0.0),
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

seg_cmap = cm.get_cmap("rainbow")

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        self.seg_pred_dir = os.path.join(self.directory, "seg_pred")
        self.vot_pred_dir = os.path.join(self.directory, "vot_pred")
        self.writer = None

    def create_summary(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.directory))
        if not os.path.exists(self.seg_pred_dir): os.mkdir(self.seg_pred_dir)
        if not os.path.exists(self.vot_pred_dir): os.mkdir(self.vot_pred_dir)
            

    def add_scalar(self, name, data, step):
        self.writer.add_scalar(name, data, step)

    def visualize_evaluator(self, epoch, confusion_matrix, name_prefix="train/"):
        Acc, Acc_ = metrics.Pixel_Accuracy(confusion_matrix)
        Acc_class, Acc_class_ = metrics.Pixel_Accuracy_Class(confusion_matrix)
        mIoU = metrics.Mean_Intersection_over_Union(confusion_matrix)
        FWIoU = metrics.Frequency_Weighted_Intersection_over_Union(confusion_matrix)
        self.writer.add_scalar(name_prefix + "mIoU", mIoU, epoch)
        self.writer.add_scalar(name_prefix + "Acc", Acc, epoch)
        self.writer.add_scalar(name_prefix + "Acc_w/o_bg", Acc_, epoch)
        self.writer.add_scalar(name_prefix + "Acc_class", Acc_class, epoch)
        self.writer.add_scalar(name_prefix + "Acc_class_w/o_bg", Acc_class_, epoch)
        self.writer.add_scalar(name_prefix + "FWIoU", FWIoU, epoch)
        # print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        return mIoU, Acc, Acc_class, FWIoU

    def visualize_seg_evaluator(self, evaluator, epoch, name_prefix="train/seg_"):
        return self.visualize_evaluator(epoch, evaluator.seg_confusion_matrix, name_prefix + "seg_")

    def visualize_vertex_evaluator(self, evaluator, epoch, name_prefix="train/"):
        return self.visualize_evaluator(epoch, evaluator.vertex_confusion_matrix, name_prefix + "vertex_")
    
    def visualize_mask_evaluator(self, evaluator, epoch, name_prefix="train/"):
        return self.visualize_evaluator(epoch, evaluator.mask_confusion_matrix, name_prefix + "mask_")


    def visualize_seg_image(self, image, output, target, epoch, i, global_step, color_map=seg_cmap):
        # image = torch.squeeze(image).permute(1, 2, 0).cpu().numpy()
        image = torch.squeeze(image).cpu().numpy()
        target = target.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        img = np.concatenate((output, target), axis=1).astype(np.int)
        img = (color_map[img]).astype(np.uint8)
        img = np.concatenate((image, img), axis=1).astype(np.uint8).transpose(2, 0, 1)
        _img = cv2.cvtColor(img.copy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.seg_pred_dir, f"seg_pred_{global_step}.png"), _img)
        self.writer.add_image("val/epoch_{}_seg_{}".format(epoch, i), img, global_step)

    def visualize_vertex_image(self, image, output, target, epoch, i, global_step, pt2d=None, visualize_landmarks=False):
        # image = torch.squeeze(image).permute(1, 2, 0).cpu().numpy()
        image = image.squeeze().cpu().numpy()
        target = target.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        output = output.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        
        # print(target.shape, output.shape)

        img = np.concatenate((output, target), axis=1)
        img = (np.arctan2(img[..., 1], img[..., 0]) * 180 / np.pi).astype(np.int64) + 180
        # img = np.concatenate((image, vertex_cmap(img)[..., :3]*255), axis=1).astype(np.uint8).transpose(2, 0, 1)

        img2 = np.uint8(vertex_cmap(img)[..., :3] * 255)

        pt2d = pt2d.astype(np.int32)
        if visualize_landmarks:
            for idx in range(pt2d.shape[0]):
                img2 = cv2.drawMarker(img2, (pt2d[idx, 0], pt2d[idx, 1]), [0, 0, 0], 0, 20, 2)

        img = np.concatenate((image.astype(np.uint8), img2), axis=1).transpose(2, 0, 1)
        _img = cv2.cvtColor(img.copy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.vot_pred_dir, f"vot_pred_{global_step}.png"), _img)
        self.writer.add_image("val/epoch_{}_vertex_{}".format(epoch, i), img, global_step)
        
    def visualize_mask_image(self, image, output, target, epoch, i, global_step):
        image = image.squeeze().cpu().numpy()
        h, w = target.shape
        target = target.squeeze().detach().cpu().unsqueeze(2).expand(h, w, 3).numpy()
        output = output.squeeze().detach().cpu().unsqueeze(2).expand(h, w, 3).numpy()
        img = np.concatenate((output, target), axis=1)
        img = np.concatenate((image, img*255), axis=1).astype(np.uint8).transpose(2, 0, 1)
        self.writer.add_image("val/epoch_{}_mask_{}".format(epoch, i), img, global_step)

    def close(self):
        self.writer.close()
