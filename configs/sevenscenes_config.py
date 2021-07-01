from configs.base_config import BaseConfig


class SevenScenesConfig(BaseConfig):
    def set_default_opts(self):
        super().set_default_opts()

        self.opt["coding_book_filename"] = "channel(24)_isomap(k3)_7scenes_chess.json"

        self.opt["dataset"] = "7scenes"
        self.opt["scene"] = "office"
        self.opt["base_dir"] = "/home/zhouhan/data/7scenes_release"
        self.opt["data_dir"] = ""


class SevenScenesLocConfig(SevenScenesConfig):
    def set_default_opts(self):
        super().set_default_opts()

        self.opt["message_prefix"] = "loc"

        self.opt["seg_decoder"] = "v1"
        self.opt["seg_loss_type"] = "embedding_v3"
        self.opt["vertex_decoder"] = "v2"
        self.opt["vertex_loss_type"] = "l1_loss"

        self.opt["use_own_nn"] = True

        self.opt["predict_full_image"] = True

        self.opt["lr"] = 0.0001
        self.opt["devices"] = "0"
        self.opt["seg_channel"] = 24
        self.opt["experiment"] = "aug1009/"
        self.opt["train_batch_size"] = 2
        self.opt["sync_bn"] = False

        self.opt["visualize_segmenation"] = False
        self.opt["visualize_voting"] = False

        self.opt["seg_k"] = 2
        self.opt["seg_loss_margin"] = 0.5

        self.opt["epochs"] = 100
        self.opt["eval_epoch_begin"] = 90
        self.opt["eval_interval"] = 5
        self.opt["vertex_loss_ratio"] = 3.0
        self.opt["val_label_filter_threshsold"] = 20

        self.opt["validation_debug"] = False
