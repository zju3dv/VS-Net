import os.path as osp
import shutil
import torch
from collections import OrderedDict
import json

class Saver(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.checkpoint_dir = cfg["checkpoint_dir"]
        self.export_dir = cfg["export_dir"]

    def save_checkpoint(self, state, is_best, filename="checkpoint.pth.tar", save_model=True):
        """Saves checkpoint to disk"""
        filename = osp.join(self.checkpoint_dir, filename)
        if save_model: torch.save(state, filename)
        if is_best:
            best_pred = state["best_pred"]
            with open(osp.join(self.export_dir, "best_pred.txt"), "w") as f:
                json.dump(best_pred, f)
            if save_model: shutil.copyfile(filename, osp.join(self.export_dir, "model_best.pth.tar"))