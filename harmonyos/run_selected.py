"""Run the explicitly selected experiments serially on a physical device."""
import argparse,json,subprocess,sys
from pathlib import Path
from validate_result import validate
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--target',required=True);p.add_argument('--run',action='append');a=p.parse_args()
runs=a.run or json.loads((ROOT/'selected_runs.json').read_text())['run_order']
for run in runs:
 for smoke in (True,False):
  existing=set((ROOT/'results').glob(run+'__*'))
  command=[sys.executable,str(ROOT/'run_device.py'),'--target',a.target,'--run',run]+(['--smoke'] if smoke else [])
  subprocess.run(command,check=True)
  created=set((ROOT/'results').glob(run+'__*'))-existing
  assert len(created)==1,'Ambiguous result directory'
  path=created.pop();acceptance=validate(path);(path/'acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')
  print('ACCEPTED',path.name,flush=True)
