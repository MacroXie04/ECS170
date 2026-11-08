"""Publish compact verified evidence, excluding raw images, weights and device logs."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
from validate_result import validate
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('result_dir',type=Path);a=p.parse_args();source=a.result_dir
accepted=validate(source);assert accepted['formal'],'Only full experiments are published'
r=json.loads((source/'result.json').read_text());job=json.loads((source/'job.json').read_text());run=job['modelId'];out=ROOT/'reports'/run;out.mkdir(parents=True,exist_ok=True)
for name in ('protocol_run.json','data_manifest.json','source_hashes.json','model_hashes.json','host_timing.json','reload.json'):
 shutil.copy2(source/name,out/name)
(out/'acceptance.json').write_text(json.dumps(accepted,indent=2)+'\n')
(out/'metrics.json').write_text(json.dumps(r,indent=2)+'\n')
(out/'device.txt').write_text((source/'device.txt').read_text())
with (out/'epochs.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(r['epochs'][0]));writer.writeheader();writer.writerows(r['epochs'])
artifacts={}
for name in ('weights_before.bin','weights_after.bin','best.ms','best_state.bin','best_state.json','test_logits.bin','reloaded_logits.bin'):
 path=source/name;artifacts[name]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(out/'artifact_hashes.json').write_text(json.dumps(artifacts,indent=2)+'\n')
rows=[]
for path in sorted((ROOT/'reports').glob('*/metrics.json')):
 m=json.loads(path.read_text());name=path.parent.name;seconds=m['elapsedMs']/1000
 rows.append(f"| {name} | {len(m['epochs'])} | {m['initialTrainLoss']:.6g} → {m['epochs'][-1]['trainEvalLoss']:.6g} | {m['bestValAccuracy']:.2%} | {m['testAccuracy']:.2%} | {seconds:.3f} s | {m['optimizerMs']/1000:.3f} s |")
text='''# Verified physical-device results

Source: PR #6, seed 2026. Each row passed full epoch/sample coverage, finite and
changed weights, and inference reload in an independent application process.
Total device time includes loading, evaluation and saving; optimizer time records
native training steps. Data-transfer time is excluded. Training CE is evaluated
on the full optimization split after each epoch, with augmentation disabled.

| Run | Epochs | Initial → final train CE | Best validation | Test | Total device time | Optimizer time |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'\n\nOnly completed, verified runs are listed. Raw models/data remain local; artifact hashes identify them.\n'
(ROOT/'reports/README.md').write_text(text)
print('PUBLISHED',out)
