import torch
from sklearn.metrics import f1_score as skl_f1_score


def f1_score(prediction_logits, targets, threshold=0.5):
    predictions = torch.sigmoid(prediction_logits)
    return f1_score_from_probs(predictions, targets, threshold)


def f1_score_from_probs(predictions, targets, threshold=0.5):
    binary_predictions = (predictions > threshold).float()
    return skl_f1_score(targets, binary_predictions, average="macro")
