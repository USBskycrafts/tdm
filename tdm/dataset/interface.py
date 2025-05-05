from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tdm.utils import load_instance


class Dataset(pl.LightningDataModule):
    def __init__(self,
                 dataset_config):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_config = dataset_config
        self.batch_size = dataset_config['batch_size']

    def setup(self, stage=None):
        self.train_dataset = load_instance(self.dataset_config['train'])
        self.val_dataset = load_instance(self.dataset_config['val'])
        self.test_dataset = load_instance(self.dataset_config['test'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          pin_memory=True,
                          persistent_workers=True,
                          num_workers=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          persistent_workers=True,
                          num_workers=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          persistent_workers=True,
                          num_workers=self.batch_size)
