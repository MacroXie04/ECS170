"""Deploy and monitor real-device PR6 runs; never execute host optimizer steps."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/data/storage/el2/base/haps/entry/files/'
BUNDLE='com.example.mnistprobe'

def main():
 p=argparse.ArgumentParser();p.add_argument('--target',required=True);p.add_argument('--run',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if ':' in a.target:raise SystemExit('Physical USB target required')
 hdc=os.environ.get('HDC',str(Path.home()/'Applications/DevEco-Studio.app/Contents/sdk/default/openharmony/toolchains/hdc'))
 def cmd(*args):
  r=subprocess.run([hdc,'-t',a.target,*args],capture_output=True,text=True,timeout=120)
  if r.returncode or '[Fail]' in r.stdout or (args[0]!='shell' and 'error:' in r.stdout.lower()):raise RuntimeError(r.stdout+r.stderr)
  return r.stdout
 def send(path,name):cmd('file','send','-b',BUNDLE,str(path),REMOTE+name)
 def recv(name,path):
  try:cmd('file','recv','-b',BUNDLE,REMOTE+name,str(path))
  except RuntimeError:return False
  return path.exists()
 def start():cmd('shell','aa force-stop '+BUNDLE);cmd('shell','aa start -a EntryAbility -b '+BUNDLE)
 r=next(r for r in json.loads((ROOT/'protocol.json').read_text())['runs'] if r['id']==a.run);c=r['config'];m=json.loads((ROOT/'data/manifest.json').read_text())[c['dataset']]
 ident=a.run+('__smoke_' if a.smoke else '__device_')+str(time.time_ns());out=ROOT/'results'/ident;out.mkdir(parents=True)
 ch,h,w=m['input_shape'];job=dict(id=ident,modelId=a.run,dataset=c['dataset'],height=h,width=w,inputChannels=ch,classes=40 if c['dataset']=='orl' else 10,batchSize=c['batch_size'],epochs=c['max_epoch'],learningRate=c['learning_rate'],optimizer=c['optimizer'],trainCount=m['optimization_samples'],valCount=m['validation_samples'],testCount=m['instructor_test_samples'],mean=m['mean'],std=m['std'],replicate=c['aug_padding_mode']=='replicate',action='smoke' if a.smoke else 'train',smokeSteps=2)
 (out/'job.json').write_text(json.dumps(job,indent=2));(out/'protocol_run.json').write_text(json.dumps(r,indent=2));(out/'data_manifest.json').write_text(json.dumps(m,indent=2))
 device=cmd('shell','param get const.product.devicetype; param get const.product.model; param get const.ohos.fullname; uname -m')
 if '2in1' not in device:raise RuntimeError('Expected a HarmonyOS computer: '+device)
 (out/'device.txt').write_text(device)
 source_files=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.ets','.cpp','.inc','.sh') and not any(part in ('build','oh_modules','.hvigor','legacy') for part in p.relative_to(ROOT).parts)]
 (out/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_files)},indent=2))
 for path in sorted((ROOT/'data').glob(c['dataset']+'_*.bin')):send(path,path.name)
 for path in sorted((ROOT/'data'/a.run).glob('*.bin')):send(path,a.run+'_'+path.name)
 model_dir=ROOT/'models'/a.run
 (out/'model_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in model_dir.glob('*.ms')},indent=2))
 for mode in ['train','eval']+(['tail'] if job['trainCount']%job['batchSize'] else []):send(model_dir/(mode+'.ms'),a.run+'_'+mode+'.ms')
 send(model_dir/'variables.json',a.run+'_variables.json');send(out/'job.json','job.json');start()
 started=time.monotonic();last=0;print('START',ident,flush=True);print('Evidence:',out,flush=True)
 while True:
  if recv(ident+'_result.json',out/'result.json'):
   result=json.loads((out/'result.json').read_text());print('RESULT',json.dumps(result),flush=True)
   (out/'device.log').write_text(cmd('shell','hilog -x -T Stage3Train'))
   if result['state'] not in ('completed','smoke_passed'):raise RuntimeError(result.get('error','Device training failed'))
   break
  if recv(ident+'_progress.json',out/'progress.json'):
   progress=json.loads((out/'progress.json').read_text())
   if len(progress['epochs'])!=last:last=len(progress['epochs']);print('EPOCH',json.dumps(progress['epochs'][-1]),flush=True)
  elapsed=int(time.monotonic()-started)
  if elapsed>30 and not cmd('shell','pidof '+BUNDLE).strip():raise RuntimeError('Training app exited before a result; inspect device crash logs')
  print('ELAPSED %02d:%02d:%02d'%(elapsed//3600,elapsed//60%60,elapsed%60),flush=True)
  time.sleep(10)
 for suffix in ['weights_before.bin','weights_after.bin','best.ms','best_state.bin','best_state.json']+(['val_logits.bin'] if a.smoke else ['test_logits.bin']):
  if not recv(ident+'_'+suffix,out/suffix):raise RuntimeError('Cannot collect '+suffix)
 job['action']='reload_smoke' if a.smoke else 'reload';(out/'reload_job.json').write_text(json.dumps(job));send(out/'reload_job.json','job.json');start()
 deadline=time.monotonic()+1200
 while not recv(ident+'_reload.json',out/'reload.json'):
  if time.monotonic()>deadline:raise TimeoutError('Reload verification')
  time.sleep(5)
 reload=json.loads((out/'reload.json').read_text());print('RELOAD',reload,flush=True)
 if not reload['passed']:raise RuntimeError('Reload verification failed')
 if not recv(ident+'_reloaded_logits.bin',out/'reloaded_logits.bin'):raise RuntimeError('Missing reload logits')
 (out/'reload_device.log').write_text(cmd('shell','hilog -x -T Stage3Train'))
 (out/'host_timing.json').write_text(json.dumps({'including_transfers':False,'monitor_and_collection_seconds':time.monotonic()-started,'device_optimizer_seconds':result['optimizerMs']/1000,'device_training_loop_seconds':result['trainingMs']/1000,'device_evaluation_seconds':result['evaluationMs']/1000,'device_total_seconds':result['elapsedMs']/1000},indent=2))
 print('VERIFIED',out,flush=True)
if __name__=='__main__':main()
