"""Persist one run's test evaluation; existing results are never overwritten."""
from pathlib import Path

import numpy as np

from local_code.base_class.result import result
from local_code.stage_3_code.Runtime import atomic_write, write_json


class Result_Saver(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = 'test'

    @property
    def metrics_path(self):
        folder = Path(self.result_destination_folder_path)
        return folder / f'{self.result_destination_file_name}_metrics.json'

    @property
    def predictions_path(self):
        folder = Path(self.result_destination_folder_path)
        return folder / f'{self.result_destination_file_name}_predictions.npz'

    def save(self):
        """self.data holds 'metrics' (a dict) and 'predictions' (named arrays).

        The metrics file is written last and marks the result as complete, so
        predictions left by an interrupted save are safely replaced.
        """
        if self.metrics_path.exists():
            raise FileExistsError(f'Refusing to overwrite {self.metrics_path}')
        atomic_write(self.predictions_path, self._write_predictions)
        write_json(self.metrics_path, self.data['metrics'])

    def _write_predictions(self, target):
        with open(target, 'wb') as handle:
            np.savez_compressed(handle, **self.data['predictions'])

    def load(self):
        raise NotImplementedError('Use Result_Loader to read saved results')
