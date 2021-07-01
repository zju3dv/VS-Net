from configs.base_config import BaseConfig


class CambridgeConfig(BaseConfig):
    def set_default_opts(self):
        super().set_default_opts()

        self.opt["color_map_filename"] = "id2colors.json"
        self.opt["coding_book_filename"] = "channel(64)_isomap(k24)_cambridge_ShopFacade.json"

        self.opt["dataset"] = "cambridge"
        self.opt["scene"] = "OldHospital_20200919_clean_2.5"
        self.opt["base_dir"] = "/home/zhouhan/data/cambridge_release"
        self.opt["eval_epoch_begin"] = 80
        self.opt["eval_interval"] = 5
        self.opt["epochs"] = 100


class CambridgeLocConfig(CambridgeConfig):
    def set_default_opts(self):
        super().set_default_opts()

        self.opt["message_prefix"] = "loc"

        self.opt["seg_decoder"] = "v1"
        self.opt["seg_loss_type"] = "embedding_v3"
        self.opt["seg_channel"] = 64
        self.opt["seg_k"] = 2
        self.opt["seg_loss_margin"] = 0.5
        self.opt["vertex_decoder"] = "v2"
        self.opt["vertex_loss_type"] = "l1_loss"

        self.opt["use_own_nn"] = True

        self.opt["lr"] = 0.0001

        self.opt["visualize_segmenation"] = False
        self.opt["visualize_voting"] = False
        self.opt["devices"] = "3"
        self.opt["train_batch_size"] = 2
        self.opt["experiment"] = "aug1024/"
        self.opt["sync_bn"] = False
        self.opt["val_label_filter_threshsold"] = 20
        self.opt["validation_debug"] = False

        self.opt["use_aug"] = True
