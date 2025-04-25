from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import CNN
import torch
from torch.utils.data import DataLoader


def eval(model: CNN, test_loader: DataLoader) -> float:
    gold_labels = []
    pred_labels = []

    for data, target in test_loader:
        output_tensor_batch = model(data)
        for i in range(len(output_tensor_batch)):
            pred_labels.append(torch.argmax(output_tensor_batch[i]).item())
            gold_labels.append(target[i].item())
    acc = accuracy_score(gold_labels, pred_labels)
    return acc
