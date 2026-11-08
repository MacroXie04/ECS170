"""Stage 2 CSV loader. The instructor's two files always remain separate."""
from pathlib import Path
import hashlib
import numpy as np
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    def __init__(self, folder):
        super().__init__('Stage 2 CSV', 'First column is the class label')
        self.folder = Path(folder)

    def load_file(self, name, classes=None):
        path = self.folder / name
        raw = np.loadtxt(path, delimiter=',', dtype=np.float32, ndmin=2)
        if raw.shape[1] < 2 or not np.isfinite(raw).all():
            raise ValueError('CSV must have labels, features, and finite values')
        if not np.equal(raw[:, 0], np.floor(raw[:, 0])).all():
            raise ValueError('Class labels must be integers')
        labels, features = raw[:, 0].astype(np.int64), raw[:, 1:]
        if classes is None:
            classes, y = np.unique(labels, return_inverse=True)
        else:
            if not np.isin(labels, classes).all():
                raise ValueError('A label is absent from the training class mapping')
            y = np.searchsorted(classes, labels)
        return {
            'X': np.ascontiguousarray(features / 255.0), 'y': y.astype(np.int64),
            'classes': np.asarray(classes),
            'source': {'file': name, 'rows': len(y), 'features': features.shape[1],
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
        }

    def load(self):
        return self.load_file('train.csv')
