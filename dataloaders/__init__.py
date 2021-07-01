from dataloaders.datasets import sevenscenes, cambridge
from torch.utils.data import DataLoader

def make_data_loader(cfg, **kwargs):
    if cfg["dataset"] == "7scenes":
        train_set = sevenscenes.SevenScenesSegmentation(cfg, split="train")
        val_set = sevenscenes.SevenScenesSegmentation(cfg, split="test")
    elif cfg["dataset"] == "cambridge":
        train_set = cambridge.CambridgeSegmentation(cfg, split="train")
        val_set = cambridge.CambridgeSegmentation(cfg, split="test")
    else:
        raise NotImplementedError
    
    train_loader = DataLoader(train_set, batch_size=cfg["train_batch_size"], shuffle=cfg["shuffle"], drop_last=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg["val_batch_size"], shuffle=False, **kwargs)
    test_loader = None
    #return train_loader, val_loader, test_loader, train_set
    return train_loader, val_loader, test_loader, val_set

