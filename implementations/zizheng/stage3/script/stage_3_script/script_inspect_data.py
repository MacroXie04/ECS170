"""Inspect one training image and its CNN logits without training."""
from pathlib import Path
import sys
import argparse
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Setting import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=list(MODELS), default='MNIST')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/stage_3_data')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    data = Dataset_Loader(args.data, args.dataset).load()
    image = data['X'][:1]
    model = MODELS[args.dataset](num_classes=len(data['classes']))
    model.eval()
    print('Images:', tuple(data['X'].shape), 'Labels:', data['classes'].tolist())
    print('First original label:', data['classes'][data['y'][0]])
    print('Pixel range:', image.min().item(), image.max().item())
    print('Batch:', tuple(image.shape), 'Logits:', tuple(model(image).shape))
    if args.show:
        plt.imshow(image[0, 0] if image.shape[1] == 1 else image[0].permute(1, 2, 0), cmap='gray')
        plt.show()


if __name__ == '__main__':
    main()
