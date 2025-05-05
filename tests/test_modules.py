import unittest

import torch
import torch.nn as nn
import torchvision
from tdm.modules.unet import UNet


class TestUNet(unittest.TestCase):
    def setUp(self):
        # Code to set up the test environment
        self.unet = UNet(
            in_ch=3,
            base_ch=64,
            ch_scales=[1, 2, 4, 8],
            num_res_blocks=2,
            num_contrasts=4,
            attn_scales=[False, False, True, False],
            cond_dim=512,
            dropout=0.1
        )
        self.cond_model = nn.Sequential(
            *list(torchvision.models.resnet34(pretrained=True).children())[:-2]
        )
        
    def test_forward(self):
        # Test the forward pass of the UNet
        x = torch.randn(32, 3, 64, 64)
        t = torch.randint(0, 1000, (32,))
        aim = torch.randint(0, 4, (32,))
        cond_features = self.cond_model(x)
        self.assertEqual(cond_features.shape, (32, 512, 2, 2), "Condition features shape mismatch")
        output = self.unet(x, t, aim, cond_features)
        self.assertEqual(output.shape, (32, 3, 64, 64), "Output shape mismatch")
        