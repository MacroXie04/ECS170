"""Export PR6 random initialization without an optimizer or a training step."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from local_code.stage_3_code.Experiment_Config import CNNConfig
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Runtime import seed_everything

def main():
 p=argparse.ArgumentParser();p.add_argument('--run');a=p.parse_args()
 out=ROOT/'models/initial';out.mkdir(parents=True,exist_ok=True)
 types={'cifar':Method_CNN_CIFAR,'mnist':Method_CNN_MNIST,'orl':Method_CNN_ORL}
 for run in json.loads((ROOT/'protocol.json').read_text())['runs']:
  if a.run and run['id']!=a.run:continue
  c=CNNConfig.from_dict(run['config']);seed_everything(run['seed']);net=types[c.dataset](c,device='cpu')
  np.savez(out/(run['id']+'.npz'),**{key:t.detach().numpy() for key,t in net.state_dict().items()})
  (out/(run['id']+'.json')).write_text(json.dumps({'parameter_count':net.parameter_count(),'seed':run['seed'],'config_fingerprint':c.fingerprint(),'host_optimizer_steps':0},indent=2)+'\n')
  print(run['id'],net.parameter_count(),flush=True)
if __name__=='__main__':main()
