'''
Multiclass evaluation: accuracy plus precision / recall / F1 in weighted, macro and micro form
'''

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Metrics(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')
        y_true = self.data['true_y']
        y_pred = self.data['pred_y']
        scores = {'accuracy': accuracy_score(y_true, y_pred)}
        for avg in ['weighted', 'macro', 'micro']:
            scores['precision_' + avg] = precision_score(y_true, y_pred, average=avg, zero_division=0)
            scores['recall_' + avg] = recall_score(y_true, y_pred, average=avg, zero_division=0)
            scores['f1_' + avg] = f1_score(y_true, y_pred, average=avg, zero_division=0)
        return scores
