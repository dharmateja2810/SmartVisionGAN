import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


#Dataset

class CTScanDataset(Dataset):
    """Dataset that creates paired LR/HR images from a folder of CT scans.

    Parameters
    ----------
    root_dir : str
        Path to root folder containing images (searched recursively).
    hr_size : int
        Spatial size of the high-resolution target (default 256).
    upscale_factor : int
        Down-scaling factor for creating the LR input (default 4).
    """

    EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

    def __init__(self, root_dir: str, hr_size: int = 256, upscale_factor: int = 4):
        self.images = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(self.EXTENSIONS):
                    self.images.append(os.path.join(root, f))

        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_size, hr_size)),
            transforms.ToTensor(),
        ])

        lr_size = hr_size // upscale_factor
        self.lr_transform = transforms.Compose([
            transforms.Resize(
                (lr_size, lr_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        hr = self.hr_transform(img)
        lr = self.lr_transform(img)
        return lr, hr

    def get_image_path(self, idx):
        """Return the file path of the image at `idx`."""
        return self.images[idx]



#  Generator
class ResidualBlock(nn.Module):
    """Conv -> BN -> PReLU -> Conv -> BN  with skip connection."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """SRGAN Generator - maps 64*64 LR -> 256*256 HR (4* upscale).

    Architecture
    ------------
    1. Initial 9*9 conv -> PReLU  (shallow features)
    2. 16 Residual Blocks          (deep features)
    3. Post-residual conv + BN + global skip
    4. 2* Sub-Pixel upsample * 2  (4* total)
    5. Final 9*9 conv -> sigmoid    (RGB output in [0,1])
    """

    def __init__(self, n_residual_blocks: int = 16):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4),
            nn.PReLU(),
        )

        self.residuals = nn.Sequential(
            *[ResidualBlock(64) for _ in range(n_residual_blocks)]
        )

        self.convblock = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

        self.final = nn.Conv2d(64, 3, 9, 1, 4)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial          # global skip
        x = self.upsample(x)
        return torch.sigmoid(self.final(x))



#  Discriminator

class Discriminator(nn.Module):
    """VGG-style discriminator - classifies 256*256 images as real/fake."""

    @staticmethod
    def _block(in_c, out_c, stride):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            self._block(64, 64, 2),
            self._block(64, 128, 1),
            self._block(128, 128, 2),
            self._block(128, 256, 1),
            self._block(256, 256, 2),
            self._block(256, 512, 1),
            self._block(512, 512, 2),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x).view(x.size(0)))



#  VGG Feature Extractor (for perceptual loss)

class VGGFeatureExtractor(nn.Module):
    """Extracts features from VGG19 up to conv5_4 (layer 36).

    Weights are frozen - used only for computing perceptual loss.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg19
        vgg = vgg19(weights="IMAGENET1K_V1")
        self.features = vgg.features[:36].eval()
        for p in self.features.parameters():
            p.requires_grad = False

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, x):
        return self.features(self.normalize(x))
