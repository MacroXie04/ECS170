"""Multiclass evaluation without binary averaging."""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluate(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return dict(accuracy=float(accuracy_score(y_true, y_pred)), precision_macro=float(p),
                recall_macro=float(r), f1_macro=float(f), precision_weighted=float(wp),
                recall_weighted=float(wr), f1_weighted=float(wf),
                confusion_matrix=confusion_matrix(y_true,y_pred,labels=list(range(10))).tolist())
