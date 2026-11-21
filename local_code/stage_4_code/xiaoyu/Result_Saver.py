import pickle
from pathlib import Path
from local_code.base_class.result import result


class Result_Saver(result):
    def save(self):
        folder=Path(self.result_destination_folder_path)
        folder.mkdir(parents=True,exist_ok=True)
        filename=folder/self.result_destination_file_name
        with filename.open('wb') as f:
            pickle.dump(self.data,f)

    def load(self):
        filename=Path(self.result_destination_folder_path)/self.result_destination_file_name
        with filename.open('rb') as f:
            return pickle.load(f)
