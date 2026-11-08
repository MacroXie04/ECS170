"""Host orchestration only. All training, evaluation and inference run inside HarmonyOS."""
from pathlib import Path
import subprocess,json,time,sys
ROOT=Path(__file__).resolve().parent
HDC=Path.home()/'Applications/DevEco-Studio.app/Contents/sdk/default/openharmony/toolchains/hdc'
BUNDLE='com.example.stage3harmony';REMOTE='/data/storage/el2/base/haps/entry/files/'
def cmd(*args):
 p=subprocess.run([str(HDC),'-t','127.0.0.1:5555',*args],capture_output=True,text=True,timeout=90)
 if p.returncode or '[Fail]' in p.stdout:raise RuntimeError(p.stdout+p.stderr)
 return p.stdout
def send(path,name=None):return cmd('file','send','-b',BUNDLE,str(path),REMOTE+(name or path.name))
def recv(name,dest):
 try:cmd('file','recv','-b',BUNDLE,REMOTE+name,str(dest));return dest.exists()
 except RuntimeError:return False
def start():
 cmd('shell','aa force-stop '+BUNDLE);cmd('shell','aa start -a EntryAbility -b '+BUNDLE)
def main():
 plans=json.loads((ROOT/'plan.json').read_text())
 if len(sys.argv)>1:plans=[c for c in plans if c['id'] in sys.argv[1:]]
 for path in sorted((ROOT/'data').glob('*.bin')):send(path)
 print('FULL_DATA_TRANSFERRED',flush=True)
 for c in plans:
  name=c['id'];out=ROOT/'results'/name;out.mkdir(exist_ok=True)
  if (out/(name+'_reload.json')).exists() and json.loads((out/(name+'_reload.json')).read_text()).get('passed'):
   print('ALREADY_VERIFIED',name,flush=True);continue
  model=ROOT/'models'/(name+'.ms')
  while not model.exists():time.sleep(2)
  send(model);(ROOT/'job.json').write_text(json.dumps(c));send(ROOT/'job.json');start()
  print('START',name,'epochs',c['epochs'],'all training samples',c['trainCount'],flush=True)
  last=-1;begin=time.monotonic()
  while True:
   if recv(name+'_result.json',out/(name+'_result.json')):
    result=json.loads((out/(name+'_result.json')).read_text())
    if result['state']!='completed':raise RuntimeError(json.dumps(result))
    break
   if recv(name+'_progress.json',out/'progress.json'):
    progress=json.loads((out/'progress.json').read_text())
    if progress['epochs'] and progress['epochs'][-1]['epoch']!=last:
     metric=progress['epochs'][-1];last=metric['epoch'];print(name,'EPOCH',json.dumps(metric),flush=True)
   if time.monotonic()-begin>14400:raise TimeoutError(name+' exceeded 4 hours')
   time.sleep(10)
  for suffix in ['result.json','weights_before.bin','weights_after.bin','weights_before.json','weights_after.json','test_logits.bin','checkpoint.ms','inference.ms']:
   if not recv(name+'_'+suffix,out/(name+'_'+suffix)):raise RuntimeError('Cannot collect '+suffix)
  print('TRAIN_COMPLETE',name,'accuracy',result['testAccuracy'],'elapsed_ms',result['elapsedMs'],flush=True)
  c['verify']=True;(ROOT/'job.json').write_text(json.dumps(c));send(ROOT/'job.json');start()
  for _ in range(360):
   if recv(name+'_reload.json',out/(name+'_reload.json')):break
   time.sleep(2)
  else:raise TimeoutError('Reload '+name)
  reload=json.loads((out/(name+'_reload.json')).read_text())
  if not reload['passed']:raise RuntimeError('Reload mismatch '+name)
  if not recv(name+'_reloaded_logits.bin',out/(name+'_reloaded_logits.bin')):raise RuntimeError('No reloaded outputs')
  (out/'device.log').write_text(cmd('shell','hilog -x -T Stage3Train'))
  print('VERIFIED_RELOAD',name,reload,flush=True)
 print('ALL_EXPERIMENTS_COMPLETE',flush=True)
if __name__=='__main__':main()
