"""Save inspectable records, label arrays, and CPU-portable model weights."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from local_code.base_class.result import result


class Result_Saver(result):
    def __init__(self, folder):
        super().__init__('Stage 3 results', 'JSON, CSV, NPZ and PyTorch checkpoint')
        self.folder = Path(folder)

    def reserve(self):
        self.folder.mkdir(parents=True, exist_ok=False)

    def write_json(self, name, data):
        (self.folder / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

    def save(self):
        record, model, history, predictions = self.data
        self.write_json('record.json', record)
        with (self.folder / 'history.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        config=record['config'], input_shape=record['input_shape'],
                        classes=record['classes'], dataset=record['dataset'],
                        preprocessing='pixels / 255; NCHW; first ORL channel'), self.folder / 'model.pt')
        np.savez_compressed(self.folder / 'predictions.npz', **predictions)
