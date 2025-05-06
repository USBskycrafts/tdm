import pathlib
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_lightning import Callback
import torchvision
from torchvision.utils import make_grid


class ImageLoggingCallback(Callback):
    def __init__(self, log_every_n_steps=5):
        self.log_every_n_steps = log_every_n_steps
        
                
            
    @rank_zero_only
    @torch.no_grad()
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.global_step % self.log_every_n_steps == 0 and \
            isinstance(trainer.logger, TensorBoardLogger):
                tensorboard = trainer.logger.experiment
                assert isinstance(outputs, torch.Tensor)
                outputs = make_grid(outputs, nrow=8, normalize=True, scale_each=True)
                tensorboard.add_image('val/generation', outputs, batch_idx)
                x, *_ = batch
                x = make_grid(x, nrow=8, normalize=True, scale_each=True)
                tensorboard.add_image('val/ground_truth', x, batch_idx)
                
class ImageMetricsCallback(Callback):
    def __init__(self, data_range=1.0, log_every_n_steps=5):
        self.log_every_n_steps = log_every_n_steps
        from torchmetrics import (
            PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,
        )
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        self.psnr = PeakSignalNoiseRatio(data_range=data_range)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex",
        )
            
    @torch.no_grad()
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.global_step % self.log_every_n_steps == 0:
            x, aim, cond, *_ = batch
            assert isinstance(outputs, torch.Tensor) 
            pl_module.log("test/psnr", self.psnr(outputs, x),
                            prog_bar=False,logger=True, sync_dist=True)
            pl_module.log("test/ssim", self.ssim(outputs, x),
                          prog_bar=False, logger=True, sync_dist=True)
            pl_module.log("test/lpips", self.lpips(outputs.repeat(1, 3, 1, 1), x.repeat(1, 3, 1, 1)),
                            prog_bar=False, logger=True, sync_dist=True)
            
    def on_test_epoch_start(self, trainer, pl_module):
        self.psnr.reset()
        self.ssim.reset() 
        self.lpips.reset()
        
    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.psnr.to(pl_module.device)
        self.ssim.to(pl_module.device)
        self.lpips.to(pl_module.device)