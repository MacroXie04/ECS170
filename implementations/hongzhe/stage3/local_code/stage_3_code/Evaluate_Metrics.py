"""Multiclass metrics; binary precision/recall/F1 do not apply to these datasets."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from local_code.base_class.evaluate import evaluate


class Evaluate_Metrics(evaluate):
    """Accuracy plus macro- and weighted-averaged precision, recall, and F1.

    All classes are always scored, including any the model never predicts, so
    averages stay comparable between runs on ORL's one-image-per-class test set.
    """
    data = None

    def __init__(self, eName=None, eDescription=None, num_classes=None):
        super().__init__(eName, eDescription)
        self.num_classes = num_classes

    def evaluate(self):
        true_y = np.asarray(self.data['true_y'])
        pred_y = np.asarray(self.data['pred_y'])
        if true_y.shape != pred_y.shape or true_y.ndim != 1:
            raise ValueError(f'Label shape mismatch: {true_y.shape} vs {pred_y.shape}')
        labels = np.arange(self.num_classes)
        metrics = {'accuracy': float(accuracy_score(true_y, pred_y)),
                   'n_samples': int(true_y.size)}
        for average in ('macro', 'weighted'):
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_y, pred_y, labels=labels, average=average, zero_division=0)
            metrics[f'precision_{average}'] = float(precision)
            metrics[f'recall_{average}'] = float(recall)
            metrics[f'f1_{average}'] = float(f1)
        metrics['confusion_matrix'] = confusion_matrix(
            true_y, pred_y, labels=labels).tolist()
        return metrics
