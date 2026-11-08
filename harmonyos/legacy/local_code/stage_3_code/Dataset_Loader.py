"""Instructor pickle archive loader; use only the trusted course archive."""
import pickle
import zipfile
from local_code.base_class.dataset import dataset
class Dataset_Loader(dataset):
 def __init__(self, archive, name):
  super().__init__(name,'Official Stage 3 train/test partitions')
  if name not in {'MNIST','ORL','CIFAR'}:raise ValueError(name)
  self.archive=archive;self.name=name
 def load(self):
  with zipfile.ZipFile(self.archive) as z:self.data=pickle.loads(z.read('stage_3_data/'+self.name))
  if set(self.data)!={'train','test'}:raise ValueError('Unexpected partitions')
  return self.data
