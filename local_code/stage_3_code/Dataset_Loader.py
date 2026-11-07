'''
Dataset loader for the stage 3 image data (MNIST / ORL / CIFAR-10 pickles).

Each pickle file holds {'train': [{'image': matrix, 'label': int}, ...], 'test': [...]}.
load() returns images as float32 arrays of shape (N, C, H, W), standardised per channel with the
TRAIN statistics, and integer labels starting at 0.
'''
import pickle
import numpy as np
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    single_channel = False     # True: keep only the first channel (ORL is gray-scale stored as 3 identical channels)
    standardize = True         # per-channel (x - mean) / std using the training set

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    @staticmethod
    def _to_chw(img, single_channel):
        a = np.asarray(img, dtype=np.float32)
        if a.ndim == 2:                       # (H, W) gray-scale -> (1, H, W)
            return a[None, :, :]
        if a.shape[-1] in (1, 3):             # (H, W, C) -> (C, H, W)
            a = a.transpose(2, 0, 1)
        if single_channel:
            a = a[:1]
        return a

    def _convert(self, items):
        X = np.stack([self._to_chw(it['image'], self.single_channel) for it in items])
        y = np.array([it['label'] for it in items], dtype=np.int64)
        return X, y

    def load(self):
        print('loading data from ' + self.dataset_source_file_name + '...')
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb') as f:
            raw = pickle.load(f)
        X_tr, y_tr = self._convert(raw['train'])
        X_te, y_te = self._convert(raw['test'])
        if X_tr.max() > 1.0:                  # 0-255 pixels -> 0-1
            X_tr, X_te = X_tr / 255.0, X_te / 255.0
        if self.standardize:
            mean = X_tr.mean(axis=(0, 2, 3), keepdims=True)
            std = X_tr.std(axis=(0, 2, 3), keepdims=True) + 1e-8
            X_tr, X_te = (X_tr - mean) / std, (X_te - mean) / std
        # ORL labels are 1..40; CrossEntropyLoss needs 0..K-1
        offset = int(y_tr.min())
        y_tr, y_te = y_tr - offset, y_te - offset
        print('  train', X_tr.shape, 'test', X_te.shape, 'classes', int(y_tr.max()) + 1)
        return {'train': {'X': X_tr.astype(np.float32), 'y': y_tr},
                'test': {'X': X_te.astype(np.float32), 'y': y_te}}
