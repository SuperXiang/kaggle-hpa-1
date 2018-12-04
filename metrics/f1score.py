import torch


def f1score(prediction_logits, targets, epsilon=1e-7):
    predictions = torch.sigmoid(prediction_logits)

    positives = predictions > 0.5
    true_positives = positives * targets

    precision = true_positives.sum(dim=1) / (positives.sum(dim=1) + epsilon)
    recall = true_positives.sum(dim=1) / targets.sum(dim=1)

    score = 2 * (precision * recall) / (precision + recall)

    return score.mean()
