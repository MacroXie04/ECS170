"""Run synthetic BN/Adam/Nesterov checks on the connected physical computer."""
import argparse,json,os,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];REMOTE='/data/storage/el2/base/haps/entry/files/';BUNDLE='com.example.mnistprobe'
p=argparse.ArgumentParser();p.add_argument('--target',required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
assert ':' not in a.target
hdc=os.environ.get('HDC',str(Path.home()/'Applications/DevEco-Studio.app/Contents/sdk/default/openharmony/toolchains/hdc'))
def cmd(*args):
 r=subprocess.run([hdc,'-t',a.target,*args],capture_output=True,text=True,check=True)
 if '[Fail]' in r.stdout:raise RuntimeError(r.stdout)
 return r.stdout
assert '2in1' in cmd('shell','param get const.product.devicetype')
def send(path,name):cmd('file','send','-b',BUNDLE,str(path),REMOTE+name)
def recv(name,path):cmd('file','recv','-b',BUNDLE,REMOTE+name,str(path))
out=ROOT/'results'/('diagnostics_'+str(time.time_ns()));out.mkdir(parents=True)
r=np.load(a.reference)
for key in ('x','dy'):
 path=out/('bn_'+key+'.bin');r[key].tofile(path);send(path,path.name)
for path in a.models.glob('*.ms'):send(path,path.name)
send(ROOT/'models/orl_kernel5_seed2026/variables.json','orl_kernel5_seed2026_variables.json')
report={}
for kind in ('bn','adam','sgd'):
 ident=out.name+'_'+kind;job=dict(id=ident,modelId='orl_kernel5_seed2026',action='diagnose_bn' if kind=='bn' else 'diagnose',optimizer=kind,learningRate=.05 if kind=='sgd' else .001)
 path=out/'job.json';path.write_text(json.dumps(job));send(path,'job.json');cmd('shell','aa force-stop '+BUNDLE);cmd('shell','aa start -a EntryAbility -b '+BUNDLE)
 deadline=time.monotonic()+90
 while True:
  try:recv(ident+'_diagnostic.json',out/(kind+'_done.json'));break
  except RuntimeError:
   if time.monotonic()>deadline:raise TimeoutError(kind)
   time.sleep(2)
 if kind=='bn':
  errors={}
  for suffix,key in [('gamma','gamma'),('beta','beta'),('moving_mean','mean'),('moving_variance','var')]:
   dest=out/('bn_'+suffix+'.bin');recv(dest.name,dest);actual=np.fromfile(dest,np.float32);error=float(np.max(np.abs(actual-r[key])));assert error<2e-5,(key,error);errors[key]=error
  report[kind]={'passed':True,'steps':3,'max_absolute_errors':errors}
 else:
  dest=out/(kind+'.json');recv('optimizer_diagnostic.json',dest);actual=np.array(json.loads(dest.read_text()));w=np.array([1.,-2.,.25]);x=np.array([.2,-.3,.4]);m=np.zeros(3);v=np.zeros(3);expected=[w.copy()]
  for step in range(1,4):
   g=2*(w@x-.5)*x+(1e-4 if kind=='adam' else 5e-4)*w
   if kind=='adam':
    m=.9*m+.1*g;v=.999*v+.001*g*g;w-=.001/step*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
   else:
    m=g if step==1 else .9*m+g;w-=.05/step*(g+.9*m)
   expected.append(w.copy())
  error=float(np.max(np.abs(actual-np.array(expected))));assert error<2e-6,(kind,error);report[kind]={'passed':True,'steps':3,'max_absolute_error':error}
print(json.dumps(report,indent=2));(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
(ROOT/'validation/native_arithmetic.json').write_text(json.dumps(report,indent=2)+'\n')
