'''
Setting for a dataset whose loader already returns the train/test partition.
Optionally holds out part of the TRAIN set as validation data (never touches the test set).
'''
from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split


class Setting_Pre_Loaded(setting):
    val_fraction = 0.0

    def __init__(self, sName=None, sDescription=None):
        super().__init__(sName, sDescription)

    def load_run_save_evaluate(self):
        data = self.dataset.load()
        if self.val_fraction > 0:
            X_tr, X_val, y_tr, y_val = train_test_split(
                data['train']['X'], data['train']['y'], test_size=self.val_fraction,
                stratify=data['train']['y'], random_state=0)
            data['train'] = {'X': X_tr, 'y': y_tr}
            data['val'] = {'X': X_val, 'y': y_val}
        self.method.data = data
        learned_result = self.method.run()
        self.result.data = learned_result
        self.result.save()
        self.evaluate.data = learned_result
        return self.evaluate.evaluate()
