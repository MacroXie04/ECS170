import pickle
from pathlib import Path
import numpy as np
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    def __init__(self, name, folder):
        super().__init__(name, 'provided image train/test partition')
        self.name=name.upper()
        self.folder=Path(folder)

    def load(self):
        with (self.folder/self.name).open('rb') as f:
            data=pickle.load(f)
        result={}
        for split in ('train', 'test'):
            X_data=np.stack([np.asarray(x['image']) for x in data[split]])
            y_data=np.asarray([x['label'] for x in data[split]], dtype=np.int64)
            if self.name=='MNIST':
                X_data=X_data[:, None, :, :]
            else:
                X_data=X_data.transpose(0, 3, 1, 2)
            if self.name=='ORL':
                X_data=X_data[:, :1]
                y_data=y_data-1
            if not np.isfinite(X_data).all() or X_data.min()<0 or X_data.max()>255:
                raise ValueError('Expected finite image values in [0, 255]')
            if y_data.min()<0 or y_data.max()>=(40 if self.name=='ORL' else 10):
                raise ValueError('Unexpected label range')
            result[split]={'X': np.ascontiguousarray(X_data, dtype=np.float32)/255.0, 'y': y_data}
        return result
