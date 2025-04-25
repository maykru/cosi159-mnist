import os
import time

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from utils import AverageMeter

from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(self, model: nn.Module):
        self._model = model

    def train(
            self,
            train_loader: DataLoader,
            epochs: int,
            lr: float,
            save_dir: str,
    ) -> None:
        optimizer = optim.SGD(params=self._model.parameters(), lr=lr)
        loss_track = AverageMeter()
        total_train_time = 0
        total_loss = 0
        self._model.train()

        print("Start training...")
        for i in range(epochs):
            tik = time.time()
            loss_track.reset()

            gold_labels = []
            pred_labels = []

            for data, target in train_loader:
                optimizer.zero_grad()
                output = self._model(data)

                for j in range(len(output)):
                    pred_labels.append(torch.argmax(output[i]).item())
                    gold_labels.append(target[i].item())

                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                loss_track.update(loss.item(), n=data.size(0))

            elapse = time.time() - tik
            total_train_time = total_train_time + elapse
            total_loss = total_loss + loss_track.avg
            train_acc = accuracy_score(gold_labels, pred_labels)
            print("Epoch: [%d/%d]; Time: %.2f; Loss: %.5f; Train Accuracy: %.2f" % (i + 1, epochs, elapse, loss_track.avg, train_acc))

        print("Training completed, saving model to %s" % save_dir)
        avg_time_epoch = total_train_time / epochs
        print("Average training time per epoch: %.4f" % avg_time_epoch)
        avg_loss_epoch = total_loss / epochs
        print("Average loss per epoch: %.4f" % avg_loss_epoch)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._model.state_dict(), os.path.join(save_dir, "mnist.pth"))
        return
