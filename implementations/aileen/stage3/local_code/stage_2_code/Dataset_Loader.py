'''
Dataset loader for the stage 2 CSV data (MNIST-style: label, then 784 pixel values)
'''

import numpy as np
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data from ' + self.dataset_source_file_name + '...')
        # each line: label,feature1,...,feature784 (no header row)
        raw = np.loadtxt(self.dataset_source_folder_path + self.dataset_source_file_name,
                         delimiter=',', dtype=np.float32)
        y = raw[:, 0].astype(np.int64)
        # pixels are 0-255; scale to [0, 1] so the MLP trains stably
        X = raw[:, 1:] / 255.0
        return {'X': X, 'y': y}
