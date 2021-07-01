from easydict import EasyDict
import numpy as np
import os
import os.path as osp
import sys
import json
import shutil


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", buffering=1)

    def delink(self):
        self.log.close()

    def writeTerminalOnly(self, message):
        self.terminal.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class BaseConfig(object):
    def __init__(self, fn=None, debug=None):
        self.opt = EasyDict({})
        self.set_default_opts()
        self.need_process_with_params = True
        if fn:
            self.load_from_json(fn)
            self.need_process_with_params = False
        else:
            self.check_opt()
        if debug != None:
            self.opt["validation_debug"] = debug

    def set_default_opts(self):
        self.fn = None
        self.opt["message_prefix"] = ""
        self.opt["message"] = ""
        self.opt["seed"] = 1

        self.opt["backbone"] = "resnet"
        self.opt["out_stride"] = 16
        self.opt["sync_bn"] = True

        self.opt["val_label_filter_threshsold"] = 20

        # seg_decoder
        self.opt["seg_decoder"] = "v1"  # "v1", "v2"
        self.opt["seg_loss_type"] = "embedding_v3"
        self.opt["seg_channel"] = 12
        self.opt["seg_loss_margin"] = 100
        self.opt["seg_k"] = 24
        self.opt["visualize_segmenation"] = True

        # vertex_decoder
        self.opt["vertex_decoder"] = None  # "v1", "v2"
        self.opt["vertex_loss_type"] = "smooth_l1_loss"
        self.opt["vertex_loss_root"] = 1
        self.opt["vertex_channel"] = 2
        self.opt["vertex_loss_ratio"] = 3.0
        self.opt["seg_loss_ratio"] = 1.0
        self.opt["visualize_voting"] = True
        self.opt["visualize_landmark"] = False

        # train
        self.opt["train"] = True
        self.opt["epochs"] = 100
        self.opt["start_epoch"] = 0

        # validation
        self.opt["eval_interval"] = 5
        self.opt["eval_epoch_begin"] = 80
        self.opt["no_val"] = False

        self.opt["use_pnp"] = False
        self.opt["save_model"] = True

        # dataloader
        self.opt["train_batch_size"] = 4
        self.opt["val_batch_size"] = 1
        self.opt["test_batch_size"] = 1
        self.opt["shuffle"] = True
        self.opt["num_workers"] = 4

        # transforms
        self.opt["use_aug"] = False

        # dataset
        self.opt["dataset"] = ""
        self.opt["scene"] = ""
        self.opt["base_dir"] = ""
        self.opt["data_dir"] = ""

        self.opt["color_map_filename"] = None

        # optimizer
        self.opt["optimizer"] = "Adam"
        self.opt["weight_decay"] = 5e-4
        self.opt["momentum"] = 0.9
        self.opt["nesterov"] = False

        # learning rate
        self.opt["lr"] = 0.0001
        self.opt["lr_scheduler"] = "poly"
        self.opt["lr_step"] = 20

        # train
        self.opt["devices"] = "3"

        self.opt["use_own_nn"] = True

        # debug config
        self.opt["validation_debug"] = False

        self.opt["critical_params"] = [
            "dataset",
            "scene",
            "train_batch_size",
            "epochs",
            "lr",
            "use_aug",
            "seg_channel",
            # "seg_loss_type",
            # "seg_loss_margin",
            # "seg_loss_ratio",
            # "seg_k"
        ]

        # resume
        self.opt["resume"] = None
        self.opt["resume_checkpoint"] = ""
        self.opt["checkname"] = None

        # experiment log
        self.opt["export_dir"] = "logs"
        self.opt["log_tb_dir"] = "logs"

        self.opt["experiment"] = "exp_"
        self.opt["checkpoint_dir"] = "ckpts"
        self.opt["best_model_name"] = "best_model.pth.tar"
        self.opt["write_json"] = True
        self.opt["log_file"] = "log.txt"

    def process_with_params(self):
        if self.need_process_with_params == False:
            return
        self.opt["dataset"] = self.opt["dataset"].replace(
            '_loc', '').replace('_seg', '')
        self.opt["message"] = "{:s}_{:s}_{:s}_{:s}".format(self.opt["dataset"],
                                                           self.opt["scene"],
                                                           self.opt["message_prefix"],
                                                           self.opt["message"])
        self.opt["experiment_dir"] = ""

        critical_params = [self.opt[key]
                           for key in self.opt["critical_params"]]
        for name, param in zip(self.opt["critical_params"], critical_params):
            self.opt["experiment_dir"] += "{:s}[{:s}]".format(name, str(param))
        self.opt["experiment_dir"] = self.opt["experiment"] + \
            self.opt["experiment_dir"]

        self.opt["export_dir"] = "{:s}_{:s}".format(
            self.opt["dataset"], self.opt["export_dir"])
        if self.opt["validation_debug"] == True:
            self.opt["export_dir"] = "debug_{:s}".format(
                self.opt["export_dir"])

        self.opt["export_dir"] = osp.join(
            self.opt["export_dir"], self.opt["experiment_dir"])
        self.opt["log_tb_dir"] = osp.join(
            self.opt["export_dir"], self.opt["log_tb_dir"])
        self.opt["log_output_dir"] = self.opt["export_dir"]
        self.opt["checkpoint_dir"] = osp.join(
            self.opt["export_dir"], self.opt["checkpoint_dir"])

        self.opt["base_dir"] = self.opt["base_dir"]
        self.opt["data_dir"] = osp.join(
            self.opt["base_dir"], self.opt["scene"], self.opt["data_dir"])

        if "embedding" not in self.opt["seg_loss_type"]:
            fn = osp.join(self.opt["data_dir"], "id2centers.json")
            if osp.exists(fn):
                self.opt["seg_channel"] = len(json.load(open(fn)))

        self.need_process_with_params = False

    def set_environmental_variables(self, use_default_stdout=False):
        self.process_with_params()
        if not osp.isdir(self.opt["export_dir"]):
            os.makedirs(self.opt["export_dir"])

        # if osp.isdir(self.opt["log_output_dir"]):
        #     shutil.rmtree(self.opt["log_output_dir"])
        if not osp.isdir(self.opt["log_output_dir"]):
            os.makedirs(self.opt["log_output_dir"])
        if not osp.isdir(self.opt["checkpoint_dir"]):
            os.makedirs(self.opt["checkpoint_dir"])
        if not osp.isdir(self.opt["log_tb_dir"]):
            os.makedirs(self.opt["log_tb_dir"])

        os.environ["CUDA_VISIBLE_DEVICES"] = self.opt.devices

        if self.opt["write_json"]:
            fn = osp.join(self.opt["log_output_dir"], "config.json")
            self.write_to_json(fn)
            fn = osp.join(self.opt["log_output_dir"], "origin_config.json")
            self.write_origin_json(fn)

        log_file = osp.join(self.opt["log_output_dir"], self.opt["log_file"])
        print("logging to {:s}".format(log_file))
        if not use_default_stdout:
            stdout = Logger(log_file)
            sys.stdout = stdout

        if self.opt["resume"] and (self.opt["resume_checkpoint"] == ""):
            self.opt["resume_checkpoint"] = osp.join(
                self.opt["checkpoint_dir"], "checkpoint-backup.pth.tar")

    def print_opt(self):
        print("Params:")
        print("----------------------------------------------------")
        for key, data in self.opt.items():
            if key not in self.opt["critical_params"]:
                print("\t{:<30s}:{:s}".format(key, str(data)))
        print("----------------------------------------------------")
        print("Critical Params:")
        for key, data in self.opt.items():
            if key in self.opt["critical_params"]:
                print("\t{:<30s}:{:s}".format(key, str(data)))
        print("----------------------------------------------------")

    def update_opt(self, jdata):
        for key in jdata.keys():
            self.opt[key] = jdata[key]

    def check_opt(self):
        if self.opt["seg_channel"] is None:
            self.opt["vertex_loss_ratio"] = 1.0

        if self.opt["checkname"] is None:
            self.opt["checkname"] = "landmarknet-"+str(self.opt["backbone"])

    def __getitem__(self, index):
        return self.opt[index]

    # forbidden change value
    def __setitem__(self, k, v):
        # self.opt[k] = v
        raise NotImplementedError

    def load_from_json(self, fn, use_default=True):
        self.fn = fn
        with open(fn, "r") as f:
            jdata = json.load(f)
        if use_default:
            self.update_opt(jdata)
        else:
            self.opt = EasyDict(jdata)
        print("load opt from: {:s}".format(fn))

    def write_to_json(self, fn):
        jdata = json.dumps(self.opt, sort_keys=True, indent=4)
        with open(fn, "w") as f:
            f.write(jdata)
        print("write opt to: {:s}".format(fn))

    def write_origin_json(self, fn):
        if self.fn == None:
            return
        with open(self.fn, "r") as fw:
            jdata = json.dumps(json.load(fw), sort_keys=True, indent=4)
            with open(fn, "w") as fr:
                fr.write(jdata)
            print("write origin json to: {:s}".format(fn))
