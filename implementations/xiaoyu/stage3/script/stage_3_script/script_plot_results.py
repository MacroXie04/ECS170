import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PROJECT_ROOT=Path(__file__).resolve().parents[2]


def plot_results():
    result_path=PROJECT_ROOT/'result/stage_3_result'
    for history_path in result_path.glob('*/*/training_history.csv'):
        with history_path.open() as f:
            rows=list(csv.DictReader(f))
        fig, axes=plt.subplots(1, 2, figsize=(9, 3))
        for ax, key, label in zip(axes, ['train_loss', 'train_accuracy'], ['Training loss', 'Training accuracy']):
            ax.plot([int(r['epoch']) for r in rows], [float(r[key]) for r in rows], marker='.')
            ax.set(xlabel='Epoch', ylabel=label, title=history_path.parent.parent.name.upper())
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(history_path.parent/'learning_curve.png', dpi=180)
        plt.close(fig)
    fig, axes=plt.subplots(1, 3, figsize=(10, 2.6))
    for ax, name in zip(axes, ['mnist', 'orl', 'cifar']):
        with (result_path/name/'baseline/training_history.csv').open() as f:
            rows=list(csv.DictReader(f))
        ax.plot([int(r['epoch']) for r in rows], [float(r['train_loss']) for r in rows], marker='.')
        ax.set(xlabel='Epoch', ylabel='Training loss', title=name.upper())
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(result_path/'baseline_loss_curves.png', dpi=200)
    plt.close(fig)
    fig, ax=plt.subplots(figsize=(10, 1.5))
    ax.axis('off')
    labels=['Input image', 'Conv + ReLU\nPool 2x2\nRepeat 2 or 3 times', 'Flatten', 'Linear 128\nReLU', 'Linear K\nClass logits']
    for i, label in enumerate(labels):
        x=0.08+i*0.21
        ax.text(x, 0.5, label, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#edf2f6', edgecolor='#506478'), transform=ax.transAxes)
        if i<4:
            ax.annotate('', xy=(x+0.135,0.5), xytext=(x+0.075,0.5), xycoords='axes fraction', arrowprops=dict(arrowstyle='->'))
    fig.savefig(result_path/'cnn_architecture.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    plot_results()
