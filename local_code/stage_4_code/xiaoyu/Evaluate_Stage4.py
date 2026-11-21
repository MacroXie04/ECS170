from sklearn.metrics import accuracy_score,precision_recall_fscore_support,classification_report,confusion_matrix
from local_code.base_class.evaluate import evaluate


class Evaluate_Stage4(evaluate):
    def evaluate(self):
        if 'pred_y' not in self.data:
            return {'train_loss':float(self.data['train_loss'])}
        y=self.data['true_y']
        pred=self.data['pred_y']
        scores={'accuracy':float(accuracy_score(y,pred))}
        for avg in ('macro','micro','weighted'):
            p,r,f1,_=precision_recall_fscore_support(y,pred,average=avg,zero_division=0)
            scores['precision_'+avg]=float(p)
            scores['recall_'+avg]=float(r)
            scores['f1_'+avg]=float(f1)
        return scores

    def report(self):
        return classification_report(self.data['true_y'],self.data['pred_y'],labels=[0,1],target_names=['negative','positive'],zero_division=0)

    def confusion(self):
        return confusion_matrix(self.data['true_y'],self.data['pred_y'],labels=[0,1])
