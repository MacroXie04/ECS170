"""Plot only verified formal device reports; no training runs on the host."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
for path in sorted((ROOT/'reports').glob('*/metrics.json')):
 r=json.loads(path.read_text());assert r['formal'] and r['state']=='completed'
 rows=r['epochs'];epochs=[v['epoch'] for v in rows]
 fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
 for metric,label,color in [('trainEvalLoss','Train (evaluation mode)','#2878b5'),('valLoss','Validation','#e07a24')]:
  axes[0].plot(epochs,[v[metric] for v in rows],label=label,color=color,lw=1.7)
 axes[0].set(xlabel='Epoch',ylabel='Cross-entropy',yscale='log');axes[0].legend(fontsize=8)
 for metric,label,color in [('trainEvalAccuracy','Train (evaluation mode)','#2878b5'),('valAccuracy','Validation','#e07a24')]:
  axes[1].plot(epochs,[100*v[metric] for v in rows],label=label,color=color,lw=1.7)
 axes[1].set(xlabel='Epoch',ylabel='Accuracy (%)',ylim=(0,102));axes[1].legend(fontsize=8)
 for ax in axes:ax.grid(alpha=.2)
 fig.suptitle(f"{path.parent.name} · HarmonyOS physical device\nTest {100*r['testAccuracy']:.2f}% · total {r['elapsedMs']/1000:.1f} s · native training steps {r['optimizerMs']/1000:.1f} s",fontsize=11)
 fig.savefig(path.parent/'curves.png',dpi=160);plt.close(fig)
 print(path.parent/'curves.png')
