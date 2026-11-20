"""Read back what Result_Saver wrote."""
from pathlib import Path

import numpy as np

from local_code.base_class.result import result
from local_code.stage_3_code.Runtime import read_json


class Result_Loader(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = 'test'

    def load(self):
        folder = Path(self.result_destination_folder_path)
        name = self.result_destination_file_name
        with np.load(folder / f'{name}_predictions.npz', allow_pickle=False) as saved:
            predictions = {key: saved[key] for key in saved.files}
        self.data = {'metrics': read_json(folder / f'{name}_metrics.json'),
                     'predictions': predictions}
        return self.data

    def save(self):
        raise NotImplementedError('Use Result_Saver to write results')
