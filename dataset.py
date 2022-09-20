
import math
import random

import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mlworkflow import PickledDataset, TransformedDataset
from tools.utils import CropCenterTransform

from torchvision import transforms


class BallSizeDataset():

    def __init__(self, path, trainMode):
        self.trainMode = trainMode

        ds = PickledDataset(path)
        sideLength = 100
        self.ds = TransformedDataset(ds, [CropCenterTransform(side_length=sideLength)])

        transformList = [
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        ]
        self.transform = transforms.Compose(transformList)

        self.goodKeys = []
        for key in self.ds.keys:
            if not math.isnan(self.ds.query_item(key)["ball_size"]) or not trainMode:
                self.goodKeys.append(key)

    def __getitem__(self, idx):

        key = self.goodKeys[idx]
        data = self.ds.query_item(key)

        oriImg = data["input_image"]
        img = Image.fromarray(oriImg)
        ballSize = data["ball_size"]

        if self.trainMode:
            a = random.uniform(-180, 180)
            s = 1
            img = transforms.functional.affine(img,
                angle = a, translate=[0, 0], scale = s, shear = 0,
                interpolation=transforms.InterpolationMode.BICUBIC,
                fill=[0.485, 0.456, 0.406])
            ballSize *= s

        img = transforms.functional.center_crop(img, 50)

        img = transforms.functional.resize(img, 224,
                interpolation=transforms.InterpolationMode.BILINEAR)

        img = self.transform(img)

        return oriImg, key, img, torch.tensor([ballSize], dtype=torch.float)

    def __len__(self):
        return len(self.goodKeys)