
import argparse
import random
import os

import torch
from torch import nn
import numpy as np

from model import BallSizeModel
from dataset import BallSizeDataset

SEED = 4212

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


def train():
    os.makedirs("output", exist_ok = True)

    learningRate = 0.0001
    numEpochs = 200
    batchSize = 4

    model = BallSizeModel().cuda()

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)
    dataset = BallSizeDataset("basketball-instants-dataset/ball_dataset_trainval.pickle", True)
    dataLoader = torch.utils.data.DataLoader(
        dataset, batch_size=batchSize, shuffle=True, num_workers=4,
        worker_init_fn=seed_worker, generator=g)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(numEpochs * 2 / 3), gamma=0.1)

    for epoch in range(numEpochs):

        totalLoss = 0
        dataIt = 0

        for _, _, imgs, ballSizes in dataLoader:

            imgs = imgs.cuda()
            ballSizes = ballSizes.cuda()

            estSize = model(imgs)

            loss = criterion(estSize, ballSizes)
            #print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss += loss.data.item()

            dataIt += 1

        avgLoss = totalLoss / dataIt
        print('epoch [{}/{}], avg loss:{:.6f}'.format(epoch + 1, numEpochs, avgLoss))

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join("output", "model_%d.pth" % (epoch + 1)))

        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ball size training script")

    args = parser.parse_args()

    train()