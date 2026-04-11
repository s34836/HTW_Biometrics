"""Shared imports and paths for all course notebooks (device, data dirs, gallery file types)."""
import os
import random
import uuid
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

# Face-verification gallery: only these extensions are loaded from disk
GALLERY_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# PyTorch — Keras/TF mapping for the Siamese tutorial:
#   Layer / Model  ->  nn.Module
#   Conv2D         ->  nn.Conv2d
#   Dense          ->  nn.Linear
#   MaxPooling2D   ->  nn.MaxPool2d
#   Flatten        ->  nn.Flatten
#   Input + Functional API  ->  forward() with tensors (no separate Input layer)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

_BASE = os.path.dirname(os.path.abspath(__file__))
POS_PATH = os.path.join(_BASE, "data", "positive")
NEG_PATH = os.path.join(_BASE, "data", "negative")
ANC_PATH = os.path.join(_BASE, "data", "anchor")


def ensure_data_dirs() -> None:
    """Create data/positive, data/negative, and data/anchor if they are missing."""
    for p in (POS_PATH, NEG_PATH, ANC_PATH):
        if not os.path.isdir(p):
            os.makedirs(p)


__all__ = [
    "os",
    "random",
    "uuid",
    "cv2",
    "np",
    "plt",
    "torch",
    "nn",
    "device",
    "GALLERY_EXTS",
    "ANC_PATH",
    "NEG_PATH",
    "POS_PATH",
    "ensure_data_dirs",
]
