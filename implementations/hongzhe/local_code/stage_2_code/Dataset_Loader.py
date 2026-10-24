"""Stage 2 loader, adapted from Jiawei Zhang's course template."""
from pathlib import Path
import numpy as np
from local_code.base_class.dataset import dataset

class Dataset_Loader(dataset):
    def __init__(self, folder):
        super().__init__('Stage 2 digits', 'Instructor-provided fixed train/test CSV files')
        self.folder = Path(folder)

    def load(self):
        data = {}
        for split, expected in [('train', 60000), ('test', 10000)]:
            raw = np.loadtxt(self.folder / f'{split}.csv', delimiter=',', dtype=np.float32)
            if raw.shape != (expected, 785):
                raise ValueError(f'Unexpected {split} shape: {raw.shape}')
            y, x = raw[:, 0], raw[:, 1:]
            if not np.isfinite(raw).all() or x.min() < 0 or x.max() > 255:
                raise ValueError('Invalid or nonfinite feature value')
            if not np.isin(y, np.arange(10)).all():
                raise ValueError('Invalid class label')
            data[split] = {'X': x / np.float32(255), 'y': y.astype(np.int64)}
        return data
