'''
Concrete SettingModule class for a specific experimental SettingModule
'''

from local_code.base_class.setting import setting


class Setting_Train_Test_Split(setting):
    test_dataset = None

    def load_run_save_evaluate(self):
        data = self.dataset.load()

        self.method.data = data

        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate()

        