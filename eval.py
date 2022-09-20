
import argparse

import numpy as np
import torch
import cv2
import math
import os

from tools.utils import PredictionsDumper
from model import BallSizeModel
from dataset import BallSizeDataset


def eval(datasetPath, modelPath, visualization):
    model = BallSizeModel()
    model.load_state_dict(torch.load(modelPath))
    model = model.cuda()
    model.eval()

    dataset = BallSizeDataset(datasetPath, False)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    os.makedirs("output", exist_ok = True)
    pd = PredictionsDumper(os.path.join("output", "predictions.json"))

    errors = []
    with torch.no_grad():
        for oriImgs, view_key, imgs, ballSizes in dataLoader:

            imgs = imgs.cuda()
            ballSizes = ballSizes.cuda()

            estSize = model(imgs)

            gt = ballSizes.item()
            prediction = estSize.item()

            error = np.abs(gt - prediction)
            if not math.isnan(gt):
                errors.append(error)

            print("Estimation error:", error)

            pd(view_key, float(prediction))

            if visualization:
                outputImg = oriImgs[0].numpy()
                cv2.circle(outputImg, (outputImg.shape[1] // 2, outputImg.shape[0] // 2),
                        round(estSize[0].item() / 2), (0, 0, 255))
                cv2.imshow("outputImg", outputImg)
                cv2.waitKey()

    if errors:
        print("MADE", np.mean(errors))
    pd.write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ball size evaluation script")
    parser.add_argument("datasetPath", help="Path to the dataset pickle file")
    parser.add_argument("modelPath", help="Path to the model")
    parser.add_argument("--visualization", action="store_true", help="Display the estimated ball diameter")

    args = parser.parse_args()

    eval(args.datasetPath, args.modelPath, args.visualization)
