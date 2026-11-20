import pickle
from pathlib import Path
from local_code.base_class.result import result


class Result_Saver(result):
    data=None
    result_destination_folder_path=None
    result_destination_file_name='predictions.pkl'

    def save(self):
        print('saving prediction results...')
        save_path=Path(self.result_destination_folder_path)
        save_path.mkdir(parents=True, exist_ok=True)
        with(save_path / self.result_destination_file_name).open('wb') as f:
            pickle.dump(self.data, f)
