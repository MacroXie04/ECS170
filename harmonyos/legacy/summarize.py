"""Independent verification and plots of device-generated results; no model training."""
from pathlib import Path
import json,csv,hashlib,re,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
plans=json.loads((ROOT/'plan.json').read_text());records=[]
for c in plans:
 name=c['id'];out=ROOT/'results'/name
 if not (out/(name+'_reload.json')).exists():continue
 r=json.loads((out/(name+'_result.json')).read_text());reload=json.loads((out/(name+'_reload.json')).read_text())
 assert r['state']=='completed' and r['trainCount']==c['trainCount'] and r['testCount']==c['testCount']
 assert r['steps']==c['epochs']*c['trainCount']//c['batchSize']
 assert len(r['epochs'])==c['epochs']+1 and r['epochs'][-1]['loss']<r['epochs'][0]['loss']
 y=np.fromfile(ROOT/'data'/(c['dataset']+'_test_labels.bin'),dtype=np.uint8)
 logits=np.fromfile(out/(name+'_test_logits.bin'),dtype='<f4').reshape(c['testCount'],c['classes'])
 new=np.fromfile(out/(name+'_reloaded_logits.bin'),dtype='<f4').reshape(logits.shape)
 assert np.isfinite(logits).all() and np.isfinite(new).all()
 error=float(np.max(np.abs(logits-new)));assert error<1e-4 and reload['passed']
 before=np.fromfile(out/(name+'_weights_before.bin'),dtype='<f4');after=np.fromfile(out/(name+'_weights_after.bin'),dtype='<f4')
 assert before.shape==after.shape and np.isfinite(before).all() and np.isfinite(after).all()
 changed=int((before!=after).sum());assert changed>0
 pred=logits.argmax(1);cm=np.zeros((c['classes'],c['classes']),np.int64);np.add.at(cm,(y,pred),1)
 tp=cm.diagonal();precision=np.divide(tp,cm.sum(0),out=np.zeros(c['classes']),where=cm.sum(0)>0);recall=np.divide(tp,cm.sum(1),out=np.zeros(c['classes']),where=cm.sum(1)>0);f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(c['classes']),where=precision+recall>0)
 x=logits.astype(np.float64);peak=x.max(1);loss=float((np.log(np.exp(x-peak[:,None]).sum(1))+peak-x[np.arange(len(y)),y]).mean());accuracy=float((pred==y).mean())
 assert abs(accuracy-r['testAccuracy'])<1e-10 and abs(loss-r['testLoss'])<1e-5
 log=(out/'device.log').read_text()
 trainpid=re.findall(r'\s(\d+)\s+\d+\s+I\s+.*'+name+r' COMPLETE',log)[-1]
 reloadpid=re.findall(r'\s(\d+)\s+\d+\s+I\s+.*'+name+r' RELOAD',log)[-1]
 assert trainpid!=reloadpid
 model=out/(name+'_inference.ms');assert model.stat().st_size>1000
 record={'id':name,'dataset':c['dataset'],'variant':c['variant'],'train_count':c['trainCount'],'test_count':c['testCount'],'epochs':c['epochs'],'optimizer_steps':r['steps'],'parameters':len(before),'changed_parameters':changed,'initial_train_loss':r['epochs'][0]['loss'],'final_train_loss':r['epochs'][-1]['loss'],'test_loss':loss,'accuracy':accuracy,'macro_precision':float(precision.mean()),'macro_recall':float(recall.mean()),'macro_f1':float(f1.mean()),'weighted_f1':float(np.average(f1,weights=cm.sum(1))),'device_elapsed_seconds':r['elapsedMs']/1000,'reload_max_error':error,'reload_bitwise_identical':bool(np.array_equal(logits,new)),'train_pid':int(trainpid),'reload_pid':int(reloadpid),'inference_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'verified':True}
 (out/'verification.json').write_text(json.dumps(record,indent=2)+'\n');np.savetxt(out/'confusion_matrix.csv',cm,fmt='%d',delimiter=',')
 records.append(record)
if '--require-all' in sys.argv:assert len(records)==9,f'Only {len(records)} completed'
if not records:print('No completed results yet');sys.exit(0)
(ROOT/'results/summary.json').write_text(json.dumps(records,indent=2)+'\n')
with (ROOT/'results/summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
for col,dataset in enumerate(['MNIST','ORL','CIFAR']):
 for rec in records:
  if rec['dataset']!=dataset:continue
  name=rec['id'];r=json.loads((ROOT/'results'/name/(name+'_result.json')).read_text());e=r['epochs']
  axes[0,col].plot([v['epoch'] for v in e],[v['loss'] for v in e],label=rec['variant'])
  axes[1,col].plot([v['epoch'] for v in e],[v['accuracy']*100 for v in e],label=rec['variant'])
 axes[0,col].set(title=dataset+' — full training set',xlabel='Epoch',ylabel='Cross-entropy loss',yscale='log');axes[1,col].set(xlabel='Epoch',ylabel='Training accuracy (%)',ylim=(0,101))
 for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend()
fig.suptitle('Stage 3: all optimization executed in HarmonyOS PC emulator',fontsize=16);fig.savefig(ROOT/'results/learning_curves.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,5),layout='constrained');bars=ax.bar([r['id'] for r in records],[100*r['accuracy'] for r in records],color=['#2563eb' if r['dataset']=='MNIST' else '#059669' if r['dataset']=='ORL' else '#9333ea' for r in records]);ax.bar_label(bars,fmt='%.2f%%',padding=3);ax.set(ylim=(0,108),ylabel='Full test-set accuracy (%)',title='Fixed final-epoch evaluation — no test-set tuning');ax.tick_params(axis='x',rotation=35);fig.savefig(ROOT/'results/test_accuracy.png',dpi=180);plt.close(fig)
print(json.dumps([{'id':r['id'],'accuracy':r['accuracy'],'macro_f1':r['macro_f1'],'verified':True} for r in records],indent=2))
