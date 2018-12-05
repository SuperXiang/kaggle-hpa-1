import torch


def f1_score(prediction_logits, targets, threshold=0.5, epsilon=1e-7):
    predictions = torch.sigmoid(prediction_logits)
    return f1_score_from_probs(predictions, targets, threshold, epsilon)


def f1_score_from_probs(predictions, targets, threshold=0.5, epsilon=1e-7):
    positives = (predictions > threshold).float()
    true_positives = positives * targets

    precision = true_positives.sum(dim=1) / (positives.sum(dim=1) + epsilon)
    recall = true_positives.sum(dim=1) / targets.sum(dim=1)

    score = 2 * (precision * recall) / (precision + recall + epsilon)

    return score.mean()
