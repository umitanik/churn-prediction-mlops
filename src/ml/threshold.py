import numpy as np
from sklearn.metrics import precision_recall_curve


def fbeta(precision, recall, beta):
    b2 = beta ** 2
    denominator = b2 * precision + recall
    return np.where(denominator > 0, (1 + b2) * precision * recall / denominator, 0.0)


def choose_threshold(y_true, y_prob, beta=2.0):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    scores = fbeta(precision[:-1], recall[:-1], beta)
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


def expected_net_benefit(y_true, y_prob, threshold, contact_cost, retained_value, success_rate):
    flagged = y_prob >= threshold
    reached = np.sum(flagged & (np.asarray(y_true) == 1))
    return success_rate * retained_value * reached - contact_cost * np.sum(flagged)
