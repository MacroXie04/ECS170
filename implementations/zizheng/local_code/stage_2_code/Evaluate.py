"""Multiclass metrics, always with an explicit complete class list."""
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from local_code.base_class.evaluate import evaluate


class Evaluate(evaluate):
    def __init__(self, classes):
        super().__init__('Multiclass evaluation', 'Accuracy and macro/weighted/micro PRF')
        self.classes = [int(c) for c in classes]

    def evaluate(self):
        true, pred = self.data['true_y'], self.data['pred_y']
        labels = list(range(len(self.classes)))
        metrics = {'accuracy': float(accuracy_score(true, pred))}
        for average in ['macro', 'weighted', 'micro']:
            p, r, f, _ = precision_recall_fscore_support(
                true, pred, labels=labels, average=average, zero_division=0)
            metrics[average] = dict(precision=float(p), recall=float(r), f1=float(f))
        metrics['per_class'] = classification_report(
            true, pred, labels=labels, target_names=[str(c) for c in self.classes],
            output_dict=True, zero_division=0)
        metrics['confusion_matrix'] = confusion_matrix(true, pred, labels=labels).tolist()
        return metrics

    def score(self, true, pred):
        self.data = {'true_y': true, 'pred_y': pred}
        return self.evaluate()
