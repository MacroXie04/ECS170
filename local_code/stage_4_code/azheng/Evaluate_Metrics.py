"""Classification metrics for the Stage 4 recurrent models."""

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class Evaluate_Metrics(evaluate):
    def evaluate(self):
        true_y = self.data["true_y"]
        pred_y = self.data["pred_y"]
        scores = {"accuracy": float(accuracy_score(true_y, pred_y))}
        for average in ("weighted", "macro", "micro"):
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_y, pred_y, average=average, zero_division=0
            )
            scores.update({
                f"precision_{average}": float(precision),
                f"recall_{average}": float(recall),
                f"f1_{average}": float(f1),
            })
        return scores
