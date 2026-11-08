"""Reject incomplete coverage, nonfinite weights, or a failed model reload."""
import argparse,array,json,math
from pathlib import Path

def validate(path):
 path=Path(path)
 if (path/'aborted.json').exists():raise ValueError('Aborted run cannot be accepted')
 r=json.loads((path/'result.json').read_text());job=json.loads((path/'job.json').read_text());reload=json.loads((path/'reload.json').read_text())
 assert r['state'] in ('completed','smoke_passed') and reload['passed']
 assert math.isfinite(reload['maxError']) and reload['maxError']<1e-4
 assert len(r['epochs'])==(job['epochs'] if r['formal'] else 3)
 if r['formal']:
  assert r['steps']==math.ceil(job['trainCount']/job['batchSize'])*job['epochs']
  assert r['seen']==job['trainCount']*job['epochs']
  assert (path/'test_logits.bin').stat().st_size==job['testCount']*job['classes']*4
 assert 1<=r['bestEpoch']<=len(r['epochs'])
 for epoch in r['epochs']:
  for name in ('trainEvalLoss','valLoss','learningRate','epochMs'):
   assert math.isfinite(epoch[name]) and epoch[name]>=0,(name,epoch[name])
 before=array.array('f');before.frombytes((path/'weights_before.bin').read_bytes())
 after=array.array('f');after.frombytes((path/'weights_after.bin').read_bytes())
 assert len(before)==len(after)>0 and all(map(math.isfinite,after))
 changed=sum(a!=b for a,b in zip(before,after));assert changed==r['changedWeights']>0
 return {'passed':True,'formal':r['formal'],'steps':r['steps'],'samples_seen':r['seen'],'finite_weights':len(after),'changed_weights':changed,'reload_max_error':reload['maxError']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('result_dir',type=Path);a=p.parse_args();report=validate(a.result_dir)
 (a.result_dir/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
