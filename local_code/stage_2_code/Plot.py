"""Plot saved records without training or reading the original test data."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_all(output):
    output = Path(output)
    records = []
    for path in sorted(output.glob('*/record.json')):
        record = json.loads(path.read_text())
        with (path.parent / 'history.csv').open() as stream:
            history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
        epochs = [row['epoch'] for row in history]
        for ax, metric in zip(axes, ['loss', 'accuracy']):
            for split in ['train', 'val']:
                key = f'{split}_{metric}'
                if key in history[0]:
                    ax.plot(epochs, [row[key] for row in history], label=split)
            if record['phase'] == 'validation':
                ax.axvline(record['best_epoch'], color='gray', ls=':', label='selected epoch')
            ax.set(xlabel='Epoch', ylabel=metric.title(), title=f"{record['name']}: {metric}")
            ax.grid(alpha=0.2)
            ax.legend()
        fig.savefig(path.parent / 'learning_curve.png', dpi=180)
        plt.close(fig)
        if record['phase'] == 'validation':
            records.append(record)
        else:
            cm = np.array(record['test']['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
            im = ax.imshow(cm, cmap='Blues')
            for i in range(len(cm)):
                for j in range(len(cm)):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7,
                            color='white' if cm[i, j] > cm.max()/2 else 'black')
            ax.set(xlabel='Predicted class', ylabel='True class', title='Final test confusion matrix',
                   xticks=range(len(cm)), yticks=range(len(cm)),
                   xticklabels=record['classes'], yticklabels=record['classes'])
            fig.colorbar(im, ax=ax)
            fig.savefig(path.parent / 'confusion_matrix.png', dpi=180)
            plt.close(fig)
    if records:
        fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
        ax.bar([r['name'] for r in records], [r['validation']['accuracy'] for r in records])
        ax.set(ylabel='Validation accuracy', ylim=(0, 1), title='Fixed validation split; no test selection')
        ax.tick_params(axis='x', rotation=20)
        fig.savefig(output / 'comparison.png', dpi=180)
        plt.close(fig)
        with (output / 'comparison.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['name', 'best_epoch', 'train_accuracy', 'validation_accuracy',
                             'validation_macro_f1', 'seconds'])
            for r in records:
                writer.writerow([r['name'], r['best_epoch'], r['train']['accuracy'],
                                 r['validation']['accuracy'], r['validation']['macro']['f1'],
                                 r['training_seconds']])
