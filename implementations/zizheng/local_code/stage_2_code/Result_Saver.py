"""Human-readable experiment records plus a CPU-portable checkpoint."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from local_code.base_class.result import result


class Result_Saver(result):
    def __init__(self, folder):
        super().__init__('Stage 2 results', 'JSON, CSV, NPZ and PyTorch checkpoint')
        self.folder = Path(folder)

    def reserve(self):
        self.folder.mkdir(parents=True, exist_ok=False)

    def write_json(self, name, data):
        (self.folder / name).write_text(json.dumps(data, indent=2, allow_nan=False),
                                       encoding='utf-8')

    def save(self):
        record, model, history, predictions = self.data
        self.write_json('record.json', record)
        with (self.folder / 'history.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'config': record['config'], 'input_dim': record['input_dim'],
                    'classes': record['classes'], 'preprocessing': 'divide_by_255'},
                   self.folder / 'model.pt')
        np.savez_compressed(self.folder / 'predictions.npz', **predictions)
