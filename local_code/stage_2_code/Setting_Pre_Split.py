'''
Setting for datasets that are already split into train and test files.
Optionally holds out part of the TRAIN set as a validation set (for hyper-parameter tuning),
so the test set can stay untouched until the final evaluation.
'''

import numpy as np
from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split


class Setting_Pre_Split(setting):
    val_fraction = 0.0   # e.g. 0.15 to carve a validation set out of the training data
    tune_mode = False    # True: score on the validation set (for tuning); False: score on the test set

    def __init__(self, sName=None, sDescription=None):
        super().__init__(sName, sDescription)
        self.test_dataset = None

    def prepare(self, sDataset, sMethod, sResult, sEvaluate, sTestDataset=None):
        super().prepare(sDataset, sMethod, sResult, sEvaluate)
        self.test_dataset = sTestDataset

    def load_run_save_evaluate(self):
        train = self.dataset.load()
        test = self.test_dataset.load()

        data = {'train': {'X': train['X'], 'y': train['y']},
                'test': {'X': test['X'], 'y': test['y']}}

        if self.val_fraction > 0:
            X_tr, X_val, y_tr, y_val = train_test_split(
                train['X'], train['y'], test_size=self.val_fraction,
                stratify=train['y'], random_state=0)
            data['train'] = {'X': X_tr, 'y': y_tr}
            data['val'] = {'X': X_val, 'y': y_val}
            data['eval_on_val'] = self.tune_mode

        self.method.data = data
        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        return self.evaluate.evaluate()
