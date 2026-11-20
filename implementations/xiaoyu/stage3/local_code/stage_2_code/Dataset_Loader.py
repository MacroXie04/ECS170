from pathlib import Path
import numpy as np
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    dataset_source_folder_path=None
    train_file_name='train.csv'
    test_file_name='test.csv'
    train_limit=None
    test_limit=None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    @staticmethod
    def _load_file(file_name, limit=None):
        if limit is not None and limit<=0:
            raise ValueError('limit must be a positive integer or None')
        if not file_name.is_file():
            raise FileNotFoundError(f'Dataset not found: {file_name}')

        data=np.loadtxt(file_name, delimiter=',', dtype=np.uint8, max_rows=limit)
        data=np.atleast_2d(data)
        if data.shape[1]!=785:
            raise ValueError(f'{file_name}: expected 785 columns, got {data.shape[1]}')
        y=data[:, 0].astype(np.int64)
        X=data[:, 1:].copy()
        if np.any(y > 9):
            raise ValueError(f'{file_name}: labels must be integers in [0, 9]')
        return {'X': X, 'y': y}

    def load(self):
        print('loading Stage 2 train/test data...')
        data_path=Path(self.dataset_source_folder_path)
        train_data=self._load_file(data_path / self.train_file_name, self.train_limit)
        test_data=self._load_file(data_path / self.test_file_name, self.test_limit)
        print(f'Training set: {len(train_data["y"])} instances, {train_data["X"].shape[1]} features')
        print(f'Testing set:  {len(test_data["y"])} instances, {test_data["X"].shape[1]} features')
        return {'train': train_data, 'test': test_data}
