from local_code.base_class.setting import setting


class Setting_Train_Test_Split(setting):
    def load_run_save_evaluate(self):
        data=self.dataset.load()
        self.method.data=data
        result=self.method.run()
        self.result.data=result
        self.result.save()
        self.evaluate.data=result
        scores=self.evaluate.evaluate_all()
        return scores, None
