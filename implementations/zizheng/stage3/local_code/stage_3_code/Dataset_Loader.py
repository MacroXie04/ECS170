"""Read the instructor's image files without mixing their train/test partitions."""
import hashlib
import pickle
from pathlib import Path
import numpy as np
import torch
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    def __init__(self, folder, name):
        super().__init__(name, 'Instructor-provided image classification data')
        if name not in ('MNIST', 'ORL', 'CIFAR'):
            raise ValueError('Unknown dataset')
        self.path, self.name = Path(folder) / name, name
        self._raw = None

    def load(self, partition='train', classes=None):
        if self._raw is None:
            with self.path.open('rb') as stream:
                self._raw = pickle.load(stream)
        samples = self._raw[partition]
        images = np.stack([s['image'] for s in samples]).astype(np.float32)
        labels = np.asarray([s['label'] for s in samples], dtype=np.int64)
        if self.name == 'MNIST':
            images = images[:, None, :, :]
        else:
            if self.name == 'ORL':
                if not (np.array_equal(images[..., 0], images[..., 1]) and
                        np.array_equal(images[..., 0], images[..., 2])):
                    raise ValueError('Expected identical ORL grayscale channels')
                images = images[..., :1]
            images = images.transpose(0, 3, 1, 2)
        if not np.isfinite(images).all() or images.min() < 0 or images.max() > 255:
            raise ValueError('Expected finite pixels in [0,255]')
        if classes is None:
            classes = np.unique(labels)
        if not np.isin(labels, classes).all():
            raise ValueError('Unknown class in test data')
        return {'X': torch.from_numpy(np.ascontiguousarray(images / 255.0)),
                'y': torch.from_numpy(np.searchsorted(classes, labels)),
                'classes': np.asarray(classes),
                'source': {'file': self.name, 'partition': partition,
                           'rows': len(labels), 'sha256': hashlib.sha256(self.path.read_bytes()).hexdigest()}}
