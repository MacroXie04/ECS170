"""Convert exported graphs using the official MindSpore Lite converter."""
import argparse,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('model_dir',type=Path);p.add_argument('--converter',type=Path,required=True);a=p.parse_args()
env=dict(os.environ);lib=a.converter.parent.parent/'lib';env['LD_LIBRARY_PATH']=str(lib)+':'+env.get('LD_LIBRARY_PATH','')
for src in sorted(a.model_dir.glob('*.mindir')):
 out=src.with_suffix('');cmd=[str(a.converter),'--fmk=MINDIR','--trainModel=true','--inputDataFormat=NCHW','--modelFile='+str(src),'--outputFile='+str(out)]
 with src.with_suffix('.conversion.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
 if not out.with_suffix('.ms').is_file():raise RuntimeError('Missing converted model')
 print('CONVERTED',src.name,flush=True)
