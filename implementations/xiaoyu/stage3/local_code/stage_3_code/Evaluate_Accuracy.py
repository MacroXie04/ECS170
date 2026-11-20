import numpy as np
from local_code.base_class.evaluate import evaluate


class Evaluate_Accuracy(evaluate):
    def __init__(self, classes):
        super().__init__('multiclass metrics', '')
        self.classes=classes

    def confusion_matrix(self):
        matrix=np.zeros((self.classes, self.classes), dtype=np.int64)
        np.add.at(matrix, (self.data['true_y'], self.data['pred_y']), 1)
        return matrix

    def evaluate_all(self):
        matrix=self.confusion_matrix()
        tp=matrix.diagonal()
        precision=np.divide(tp, matrix.sum(0), out=np.zeros(self.classes), where=matrix.sum(0)!=0)
        recall=np.divide(tp, matrix.sum(1), out=np.zeros(self.classes), where=matrix.sum(1)!=0)
        f1=np.divide(2*precision*recall, precision+recall, out=np.zeros(self.classes), where=precision+recall!=0)
        self.per_class=np.column_stack((precision, recall, f1, matrix.sum(1)))
        return {'accuracy': float(tp.sum()/matrix.sum()), 'precision_macro': float(precision.mean()),
                'recall_macro': float(recall.mean()), 'f1_macro': float(f1.mean())}

    def evaluate(self):
        return self.evaluate_all()['accuracy']
