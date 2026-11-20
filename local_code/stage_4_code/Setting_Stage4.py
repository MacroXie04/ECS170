from local_code.base_class.setting import setting


class Setting_Stage4(setting):
    def load_run_save_evaluate(self):
        loaded=self.method.data if self.method.data is not None else self.dataset.load()
        self.method.data=loaded
        result=self.method.run()
        self.result.data=result
        self.result.save()
        self.evaluate.data=result
        return self.evaluate.evaluate(),loaded
