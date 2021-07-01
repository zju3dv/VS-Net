import os
import cv2
import time
import json
import random
import inspect
import argparse
import numpy as np
from tqdm import tqdm

from dataloaders import make_data_loader
from models.sync_batchnorm.replicate import patch_replication_callback
from models.vs_net import *
from utils.loss import loss_dict
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils import utils
from torch.autograd import Variable

import os.path as osp

from configs import *

import warnings
warnings.filterwarnings("ignore")

class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Define Saver
        self.saver = Saver(cfg)
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.cfg["log_tb_dir"])
        self.summary.create_summary()

        # Define Dataloader
        kwargs = {"num_workers": cfg["num_workers"], "pin_memory": True}
        self.train_loader, self.val_loader, self.test_loader, dset = make_data_loader(
            cfg, **kwargs)

        # read landmark centers
        self.id2center = np.array(json.load(
            open(osp.join(cfg["data_dir"], "id2centers.json")))).astype(np.float64)

        self.coding_book = torch.zeros(
            (len(self.id2center), cfg["seg_channel"]), dtype=torch.float32).cuda()
        torch.nn.init.xavier_uniform(self.coding_book)

        print("coding book size = {}".format(self.coding_book.shape))
        # generate color map
        unique_label = np.arange(len(self.id2center))
        unique_label = unique_label.astype(
            np.int64) * 6364136223846793005 + 1442695040888963407
        color_map = np.zeros((unique_label.shape[0], 3), np.uint8)
        color_map[:, 0] = np.bitwise_and(unique_label, 0xff)
        color_map[:, 1] = np.bitwise_and(np.right_shift(unique_label, 4), 0xff)
        color_map[:, 2] = np.bitwise_and(np.right_shift(unique_label, 8), 0xff)
        self.color_map = np.array(color_map)

        self.coding_book = Variable(self.coding_book, requires_grad=True)

        # Define network
        model = VSNet(backbone=cfg["backbone"],
                        seg_decoder=cfg["seg_decoder"],
                        vertex_decoder=cfg["vertex_decoder"],
                        seg_channel=cfg["seg_channel"],
                        vertex_channel=cfg["vertex_channel"],
                        output_stride=cfg["out_stride"],
                        sync_bn=cfg["sync_bn"])

        train_params = [{"params": model.get_1x_lr_params(), "lr": cfg["lr"]},
                        {"params": model.get_10x_lr_params(),
                         "lr": cfg["lr"] * 10},
                        {"params": self.coding_book, "lr": cfg["lr"] * 10}
                        ]

        # Define Optimizer
        if cfg["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(train_params, momentum=cfg["momentum"],
                                        weight_decay=cfg["weight_decay"], nesterov=cfg["nesterov"])
        elif cfg["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(train_params, lr=cfg["lr"],
                                         weight_decay=cfg["weight_decay"], amsgrad=True)
        else:
            raise NotImplementedError

        # Define Criterion
        self.seg_criterion = loss_dict[cfg["seg_loss_type"]]
        self.vertex_criterion = loss_dict[cfg["vertex_loss_type"]]
        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(
            self.coding_book.shape[0], cfg["vertex_channel"])

        # Define lr scheduler
        self.scheduler = LR_Scheduler(mode=cfg["lr_scheduler"], base_lr=cfg["lr"],
                                      num_epochs=cfg["epochs"], iters_per_epoch=len(
                                          self.train_loader),
                                      lr_step=cfg["lr_step"])

        self.model = torch.nn.DataParallel(self.model)
        patch_replication_callback(self.model)
        self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = {"mIoU": 0.0, "Acc": 0.0, "Acc": 0.0,
                          "FWIoU": 0.0, "translation_median": 1000}
        if cfg["resume"] is not None and cfg["resume"] == True:
            print(os.path.isfile(cfg["resume_checkpoint"]))
            if not os.path.isfile(cfg["resume_checkpoint"]):
                raise RuntimeError("=> no checkpoint found at {}" .format(
                    cfg["resume_checkpoint"]))
            checkpoint = torch.load(cfg["resume_checkpoint"])
            cfg.opt["start_epoch"] = checkpoint["epoch"] - 1
            state_dict = checkpoint["state_dict"]
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "mask_decoder" in k: continue
                new_state_dict[k] = v
            self.model.module.load_state_dict(new_state_dict)
            # self.coding_book.load_state_dict(checkpoint["coding_book"])
            if not cfg["ft"]:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.best_pred = checkpoint["best_pred"]
            print("coding book shape:", checkpoint["coding_book"].shape)
            if "coding_book" in checkpoint.keys():
                self.coding_book = checkpoint["coding_book"]
            else:
                print("Alert! coding book does not exist in the checkpoint")
            print("=> loaded checkpoint {} (epoch {})"
                  .format(cfg["resume"], checkpoint["epoch"]))
            
            # self.saver.save_checkpoint({
            #     "epoch": checkpoint["epoch"],
            #     "state_dict": self.model.module.state_dict(),
            #     "optimizer": self.optimizer.state_dict(),
            #     "best_pred": self.best_pred,
            #     "coding_book": self.coding_book
            # }, False, filename="checkpoint-backup.pth.tar", save_model=self.cfg["save_model"])



def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Landmark Segmentation Training")
    parser.add_argument("--dataset", type=str,
                        choices=["7scenes_loc", "cambridge_loc"], help="experiment config file")
    parser.add_argument("--scene", type=str, default="",
                        help="experiment scene")
    parser.add_argument("--gpu-id", type=str, default="",
                        help="experiment gpu id")
    parser.add_argument("--use-aug", type=str, default="",
                        choices=["", "true", "false"], help="experiment use aug")
    parser.add_argument("--config", type=str, default=None,
                        help="experiment config file")
    parser.add_argument("--debug", type=str, default="",
                        choices=["", "true", "false"], help="debug")
    parser.add_argument("--resume", type=str, default="true",
                        choices=["", "true", "false"], help="resume")
    args = parser.parse_args()

    debug = None

    if args.debug != "":
        debug = (args.debug == "true")

    if args.dataset == "7scenes_loc":
        cfg = SevenScenesLocConfig(args.config, debug)
    elif args.dataset == "cambridge_loc":
        cfg = CambridgeLocConfig(args.config, debug)
    if args.scene != "":
        cfg.opt["scene"] = args.scene
    if args.gpu_id != "":
        cfg.opt["devices"] = args.gpu_id
    if args.use_aug == "true":
        cfg.opt["use_aug"] = True
    if args.resume == "true":
        cfg.opt["resume"] = True
        cfg.opt["resume_checkpoint"] = cfg["export_dir"] + \
            '/ckpts/checkpoint-backup.pth.tar'
    cfg.print_opt()
    cfg.set_environmental_variables()

    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    trainer = Trainer(cfg)
    print("Starting Epoch:", trainer.cfg["start_epoch"])
    print("Total Epoches:", trainer.cfg["epochs"])


if __name__ == "__main__":
    main()
