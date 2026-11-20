from local_code.base_class.evaluate import evaluate
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)


class Evaluate_Accuracy(evaluate):
    data=None

    def evaluate(self):
        return float(accuracy_score(self.data['true_y'], self.data['pred_y']))

    def evaluate_all(self):
        y_true, y_pred=self.data['true_y'], self.data['pred_y']
        result={'accuracy': self.evaluate()}
        for average in ('macro', 'micro', 'weighted'):
            p, r, f1, _=precision_recall_fscore_support(
                y_true, y_pred, average=average, zero_division=0)
            result[f'precision_{average}']=float(p)
            result[f'recall_{average}']=float(r)
            result[f'f1_{average}']=float(f1)
        return result

    def detailed_report(self):
        return classification_report(self.data['true_y'], self.data['pred_y'],
                                     labels=list(range(10)), zero_division=0)

    def confusion_matrix(self):
        return confusion_matrix(self.data['true_y'], self.data['pred_y'],
                                labels=list(range(10)))
