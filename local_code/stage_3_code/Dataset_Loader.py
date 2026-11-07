"""Load the instructor's Stage 3 pickles and serve them through PyTorch.

Images stay uint8 in memory as NCHW tensors. Scaling, normalization, and any
training augmentation happen one batch at a time.
"""
from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from local_code.base_class.dataset import dataset
from local_code.stage_3_code.Runtime import sha256_array, sha256_file


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    file_name: str
    train_size: int
    test_size: int
    source_shape: tuple
    input_shape: tuple  # (channels, height, width) as the model receives it
    num_classes: int
    label_offset: int  # subtracted from the file's labels to reach 0..K-1
    validation_size: int
    sha256: str


DATASET_SPECS = {
    'mnist': DatasetSpec('mnist', 'MNIST', 60000, 10000, (28, 28), (1, 28, 28), 10, 0,
                         6000, 'c328558117745723fe4e49dd7917f00509d99f7a1ec3f8e0c0036e78b9476472'),
    # One validation image per person leaves 8 training images for each of the 40.
    'orl': DatasetSpec('orl', 'ORL', 360, 40, (112, 92, 3), (1, 112, 92), 40, 1,
                       40, '94a49cfe24429ccb7caba3439924a30cf5dfbe5fbfa003b7d4edb70d17a7a271'),
    'cifar': DatasetSpec('cifar', 'CIFAR', 50000, 10000, (32, 32, 3), (3, 32, 32), 10, 0,
                         5000, '04178bcd1198435941ecf94037f09d87f5f9a51ee65bf72b6a497bbfe1c825d4'),
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Pickle can run arbitrary code, so allow only what these files use.

    The instructor's files reference exactly three numpy globals.
    """
    _ALLOWED = {
        ('numpy.core.multiarray', '_reconstruct'): ('numpy._core.multiarray', '_reconstruct'),
        ('numpy._core.multiarray', '_reconstruct'): ('numpy._core.multiarray', '_reconstruct'),
        ('numpy', 'ndarray'): ('numpy', 'ndarray'),
        ('numpy', 'dtype'): ('numpy', 'dtype'),
    }

    def find_class(self, module, name):
        target = self._ALLOWED.get((module, name))
        if target is None:
            raise pickle.UnpicklingError(f'Blocked pickle global: {module}.{name}')
        return super().find_class(*target)


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None, folder=None):
        super().__init__(dName, dDescription)
        self.spec = DATASET_SPECS[dName]
        self.dataset_source_folder_path = str(folder) if folder else None
        self.dataset_source_file_name = self.spec.file_name

    @property
    def source_path(self):
        return Path(self.dataset_source_folder_path) / self.dataset_source_file_name

    def verify_checksum(self):
        actual = sha256_file(self.source_path)
        if actual != self.spec.sha256:
            raise ValueError(f'{self.source_path} is not the instructor file: {actual}')
        return actual

    def load(self, splits=('train', 'test')):
        """Return {split: {'X': uint8 NCHW, 'y': int64 zero-based}}.

        Training code requests splits=('train',), so test images and labels are
        dropped here and never reach the training process's data structures.
        """
        self.verify_checksum()
        with open(self.source_path, 'rb') as handle:
            raw = _NumpyOnlyUnpickler(handle).load()
        expected = {'train': self.spec.train_size, 'test': self.spec.test_size}
        self.data = {split: self._convert(raw[split], split, expected[split])
                     for split in splits}
        return self.data

    def _convert(self, instances, split, expected_count):
        if len(instances) != expected_count:
            raise ValueError(f'{split}: expected {expected_count} instances, '
                             f'found {len(instances)}')
        images = np.stack([instance['image'] for instance in instances])
        labels = np.array([instance['label'] for instance in instances], dtype=np.int64)
        if images.shape[1:] != self.spec.source_shape:
            raise ValueError(f'{split}: unexpected image shape {images.shape[1:]}')
        return {'X': self._to_uint8_nchw(images),
                'y': self._to_zero_based(labels)}

    def _to_uint8_nchw(self, images):
        # MNIST is stored as float64 whole numbers; the cast must lose nothing.
        if images.dtype != np.uint8:
            converted = images.astype(np.uint8)
            if not np.array_equal(converted, images):
                raise ValueError('Pixels are not whole numbers in 0..255')
            images = converted
        if images.ndim == 3:  # MNIST: N, H, W
            return np.ascontiguousarray(images[:, None])
        if self.spec.input_shape[0] == 1:  # ORL: identical R, G, B channels
            if not (np.array_equal(images[..., 0], images[..., 1])
                    and np.array_equal(images[..., 1], images[..., 2])):
                raise ValueError('Expected identical R, G, and B channels')
            return np.ascontiguousarray(images[..., 0][:, None])
        return np.ascontiguousarray(images.transpose(0, 3, 1, 2))  # NHWC -> NCHW

    def _to_zero_based(self, labels):
        shifted = labels - self.spec.label_offset
        if shifted.min() != 0 or shifted.max() != self.spec.num_classes - 1:
            raise ValueError(f'Labels out of range: {labels.min()}..{labels.max()}')
        return shifted


def make_train_validation_split(labels, validation_size, seed):
    """Stratified hold-out from the instructor's training set only."""
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(indices, test_size=validation_size,
                                          random_state=seed, shuffle=True, stratify=labels)
    return np.sort(train_idx), np.sort(val_idx)


def split_summary(labels, train_idx, val_idx, spec, seed):
    return {
        'method': 'stratified hold-out from the instructor training set '
                  '(sklearn train_test_split)',
        'split_seed': seed,
        'instructor_train_samples': int(len(labels)),
        'optimization_samples': int(len(train_idx)),
        'validation_samples': int(len(val_idx)),
        'instructor_test_samples': spec.test_size,
        'optimization_class_counts': np.bincount(labels[train_idx],
                                                 minlength=spec.num_classes).tolist(),
        'validation_class_counts': np.bincount(labels[val_idx],
                                               minlength=spec.num_classes).tolist(),
        'optimization_indices_sha256': sha256_array(train_idx),
        'validation_indices_sha256': sha256_array(val_idx),
    }


def compute_channel_stats(images):
    """Per-channel mean and std in [0, 1] units, from the given images only."""
    values = images.astype(np.float64) / 255.0
    mean = values.mean(axis=(0, 2, 3))
    std = values.std(axis=(0, 2, 3))
    return mean.tolist(), std.tolist()


class BatchAugmentation:
    """Random shift and horizontal flip, vectorized over a whole batch.

    A dedicated generator keeps this reproducible and restorable on resume.
    """

    def __init__(self, shift=0, hflip=False, padding_mode='constant'):
        self.shift = shift
        self.hflip = hflip
        self.padding_mode = padding_mode

    @property
    def active(self):
        return bool(self.shift or self.hflip)

    def __call__(self, images, generator):
        if self.shift:
            images = self._random_shift(images, generator)
        if self.hflip:
            flip = torch.rand(images.shape[0], generator=generator) < 0.5
            images = torch.where(flip[:, None, None, None], images.flip(-1), images)
        return images

    def _random_shift(self, images, generator):
        count, channels, height, width = images.shape
        pad = self.shift
        padded = F.pad(images, (pad, pad, pad, pad), mode=self.padding_mode)
        top = torch.randint(0, 2 * pad + 1, (count,), generator=generator)
        left = torch.randint(0, 2 * pad + 1, (count,), generator=generator)
        rows = top[:, None] + torch.arange(height)
        columns = left[:, None] + torch.arange(width)
        return padded[torch.arange(count)[:, None, None, None],
                      torch.arange(channels)[None, :, None, None],
                      rows[:, None, :, None],
                      columns[:, None, None, :]]


class ImageTensorDataset(Dataset):
    """Map-style dataset that scales, augments, and normalizes per batch.

    DataLoader calls __getitems__ with a whole batch of indices, which avoids
    Python work per image. Pass an augmentation for training data only.
    """

    def __init__(self, images, labels, mean, std, augmentation=None, generator=None):
        self.images = torch.from_numpy(np.ascontiguousarray(images))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels))
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
        self.augmentation = augmentation
        self.generator = generator

    def __len__(self):
        return self.labels.numel()

    def __getitem__(self, index):
        images, labels = self.__getitems__([index])
        return images[0], labels[0]

    def __getitems__(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long)
        images = self.images[indices].to(torch.float32)
        if self.augmentation is not None:
            images = self.augmentation(images, self.generator)
        images = (images / 255.0 - self.mean) / self.std
        return images, self.labels[indices]


def _batch_is_ready(batch):
    return batch


def make_train_loader(images, labels, mean, std, config, loader_generator, augment_generator):
    augmentation = BatchAugmentation(config.aug_shift, config.aug_hflip,
                                     config.aug_padding_mode)
    data = ImageTensorDataset(images, labels, mean, std,
                              augmentation if augmentation.active else None,
                              augment_generator)
    return DataLoader(data, batch_size=config.batch_size, shuffle=True,
                      generator=loader_generator, collate_fn=_batch_is_ready)


def make_eval_loader(images, labels, mean, std, batch_size):
    data = ImageTensorDataset(images, labels, mean, std)
    return DataLoader(data, batch_size=batch_size, shuffle=False,
                      collate_fn=_batch_is_ready)
