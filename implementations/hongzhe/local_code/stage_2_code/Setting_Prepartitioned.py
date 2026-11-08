"""Preserve the provided split; no new split and no cross-validation."""
from local_code.base_class.setting import setting
from local_code.stage_2_code.Evaluate_Multiclass import evaluate

class Setting_Prepartitioned(setting):
    def __init__(self,loaded_data,model):
        super().__init__('fixed split','Train on all 60000; evaluate on the fixed 10000')
        self.loaded_data=loaded_data
        self.method=model

    def load_run_save_evaluate(self):
        self.method.data=self.loaded_data
        result=self.method.run()
        result['metrics']=evaluate(result['true_y'],result['pred_y'])
        return result
