from torchvision import *
import torch

Celeba_dataset = datasets.CelebA('./datasets/',download=True, transform=None)