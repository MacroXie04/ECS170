"""Course-template adapter. Model optimization is delegated to the HarmonyOS app."""
from pathlib import Path
import subprocess,json
from local_code.base_class.method import method
class Method_CNN(method):
 dataset_name=''
 def __init__(self,variant='baseline'):
  super().__init__('HarmonyOS CNN '+variant,'MindSpore Lite CPU training on PC/2in1 emulator')
  if variant not in {'baseline','wide','deep'}:raise ValueError(variant)
  self.variant=variant
 def run(self,trainData=None,trainLabel=None,testData=None):
  if any(x is not None for x in [trainData,trainLabel,testData]):
   raise ValueError('Use prepared complete official dataset files; arbitrary in-memory subsets are not supported.')
  root=Path(__file__).resolve().parents[2]
  run_id=self.dataset_name.lower()+'_'+self.variant
  subprocess.run(['python3',str(root/'run_experiments.py'),run_id],check=True)
  return json.loads((root/'results'/run_id/(run_id+'_result.json')).read_text())
class Method_CNN_MNIST(Method_CNN):dataset_name='MNIST'
class Method_CNN_ORL(Method_CNN):dataset_name='ORL'
class Method_CNN_CIFAR(Method_CNN):dataset_name='CIFAR'
