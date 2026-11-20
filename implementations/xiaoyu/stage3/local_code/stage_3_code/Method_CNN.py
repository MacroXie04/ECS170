import time
import numpy as np
import torch
from torch import nn
from local_code.base_class.method import method


class Method_CNN(method, nn.Module):
    def __init__(self, name, shape, classes, channels, kernel_size=3, pooling='max',
                 hidden_dim=128, max_epoch=5, learning_rate=0.001, batch_size=256,
                 device='auto', seed=2):
        method.__init__(self, name, 'CNN image classifier')
        nn.Module.__init__(self)
        if device=='auto':
            device='cuda' if torch.cuda.is_available() else 'cpu'
        self.device=torch.device(device)
        self.max_epoch=max_epoch
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.seed=seed
        layers=[]
        in_channels=shape[0]
        for n in channels:
            layers.extend([nn.Conv2d(in_channels, n, kernel_size, padding=kernel_size//2), nn.ReLU(),
                           nn.MaxPool2d(2) if pooling=='max' else nn.AvgPool2d(2)])
            in_channels=n
        self.features=nn.Sequential(*layers)
        with torch.no_grad():
            n=self.features(torch.zeros(1, *shape)).numel()
        self.classifier=nn.Sequential(nn.Flatten(), nn.Linear(n, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, classes))
        self.to(self.device)
        self.history=[]

    def forward(self, X_data):
        return self.classifier(self.features(X_data))

    def fit(self, X, y):
        X_data=torch.from_numpy(X).to(self.device)
        y_data=torch.from_numpy(y).to(self.device)
        optimizer=torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        generator=torch.Generator().manual_seed(self.seed)
        for epoch in range(1, self.max_epoch+1):
            start=time.perf_counter()
            self.train()
            order=torch.randperm(len(y), generator=generator)
            loss_sum, correct_num=0.0, 0
            for index in order.split(self.batch_size):
                index=index.to(self.device)
                output=self(X_data[index])
                loss=nn.functional.cross_entropy(output, y_data[index])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum+=loss.item()*len(index)
                correct_num+=(output.argmax(1)==y_data[index]).sum().item()
            row={'epoch': epoch, 'train_loss': loss_sum/len(y), 'train_accuracy': correct_num/len(y),
                 'seconds': time.perf_counter()-start}
            self.history.append(row)
            print(f'{self.method_name} Epoch {epoch}/{self.max_epoch}: {row}', flush=True)
        return self.history

    def test(self, X):
        self.eval()
        y_pred=[]
        with torch.inference_mode():
            for start in range(0, len(X), self.batch_size):
                X_batch=torch.from_numpy(X[start:start+self.batch_size]).to(self.device)
                y_pred.append(self(X_batch).argmax(1).cpu().numpy())
        return np.concatenate(y_pred)

    def run(self):
        self.fit(self.data['train']['X'], self.data['train']['y'])
        return {'pred_y': self.test(self.data['test']['X']), 'true_y': self.data['test']['y']}
