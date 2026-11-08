from pathlib import Path
import subprocess,json,shutil
r=Path(__file__).resolve().parent
probe=Path.home()/'Developer/harmony-mnist-probe'
for c in json.loads((r/'plan.json').read_text()):
 name=c['id']
 cmd=['docker','run','--rm','--platform','linux/amd64','-v',str(probe)+':/work','-w','/work','-e','LD_LIBRARY_PATH=/work/mindspore-lite-2.7.0-linux-x64/tools/converter/lib','ubuntu:22.04','/work/mindspore-lite-2.7.0-linux-x64/tools/converter/converter/converter_lite','--fmk=MINDIR','--trainModel=true','--modelFile=/work/stage3_models/'+name+'.mindir','--outputFile=/work/stage3_models/'+name,'--inputDataFormat=NCHW']
 with (r/'models'/f'{name}.conversion.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 for ext in ['.mindir','.ms']:shutil.copy2(probe/'stage3_models'/(name+ext),r/'models'/(name+ext))
 print('CONVERTED',name,flush=True)
