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
        # No header: the first row is a real example, not column names.
        raw = np.loadtxt(path, delimiter=',', dtype=np.float32, ndmin=2)
        if raw.shape[1] < 2 or not np.isfinite(raw).all():
            raise ValueError('CSV must have labels, features, and finite values')
        if not np.equal(raw, np.floor(raw)).all():
            raise ValueError('The supplied data specification requires integers')
        labels, features = raw[:, 0].astype(np.int64), raw[:, 1:]
        if features.min() < 0 or features.max() > 255:
            raise ValueError('Expected features in the documented 0..255 range')
        if classes is None:
            classes = np.unique(labels)
        mapping = {int(label): i for i, label in enumerate(classes)}
        if not set(labels).issubset(mapping):
            raise ValueError('A label is absent from the training class mapping')
        y = np.array([mapping[int(v)] for v in labels], dtype=np.int64)
        return {
            'X': np.ascontiguousarray(features / 255.0), 'y': y,
            'classes': np.asarray(classes),
            'source': {'file': name, 'rows': len(y), 'features': features.shape[1],
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
        }

    def load(self):
        # Selection runs load train.csv only. test.csv is loaded in final mode.
        return self.load_file('train.csv')
