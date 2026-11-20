import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from local_code.base_class.method import method


class Method_MLP(method, nn.Module):
    def __init__(self, mName, mDescription, hidden_dims=(128,), max_epoch=10,
                 learning_rate=0.001, batch_size=256, optimizer_name='adam',
                 loss_name='cross_entropy', device='auto'):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.max_epoch=max_epoch
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.optimizer_name=optimizer_name.lower()
        self.loss_name=loss_name.lower()
        self.device=self._choose_device(device)
        self.history=[]


        sizes=[784, *hidden_dims, 10]
        layers=[]
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:
                layers.append(nn.ReLU())
        self.network=nn.Sequential(*layers)

        self.to(self.device)

    @staticmethod
    def _choose_device(choice):
        if choice=='auto':
            if torch.cuda.is_available():
                choice='cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                choice='mps'
            else:
                choice='cpu'
        device=torch.device(choice)
        if device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available')
        if device.type=='mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS was requested but is not available')
        return device

    def forward(self, x):
        return self.network(x)

    def _loss_function(self, output, y_data):
        if self.loss_name=='cross_entropy':
            return nn.functional.cross_entropy(output, y_data)
        if self.loss_name=='nll':
            return nn.functional.nll_loss(nn.functional.log_softmax(output, dim=1), y_data)
        if self.loss_name=='mse':

            y_batch=nn.functional.one_hot(y_data, num_classes=10).float()
            return nn.functional.mse_loss(nn.functional.softmax(output, dim=1), y_batch)
        raise ValueError(f'Unknown loss: {self.loss_name}')

    def fit(self, X, y):


        X_data=torch.from_numpy(np.asarray(X, dtype=np.float32)) / 255.0
        y_data=torch.from_numpy(np.asarray(y, dtype=np.int64))
        data_loader=DataLoader(TensorDataset(X_data, y_data), batch_size=self.batch_size,
                            shuffle=True, num_workers=0)
        if self.optimizer_name=='adam':
            optimizer=torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name=='sgd':
            optimizer=torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f'Unknown optimizer: {self.optimizer_name}')

        self.history=[]
        for epoch in range(1, self.max_epoch+1):
            self.train()
            loss_sum, correct_num, total_num=0.0, 0, 0
            for X_batch, y_batch in data_loader:
                X_batch=X_batch.to(self.device)
                y_batch=y_batch.to(self.device)
                output=self.forward(X_batch)
                loss=self._loss_function(output, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_size=len(y_batch)
                loss_sum+=loss.item()*batch_size
                correct_num+=(output.argmax(dim=1)==y_batch).sum().item()
                total_num+=batch_size
            result={'epoch': epoch, 'train_loss': loss_sum/total_num,
                       'train_accuracy': correct_num/total_num}
            self.history.append(result)
            print(f'Epoch {epoch:>3}/{self.max_epoch} | '
                  f'Loss {result["train_loss"]:.4f} | '
                  f'Training accuracy {result["train_accuracy"]:.4f}')
        return self.history

    def test(self, X):
        self.eval()
        X_data=torch.from_numpy(np.asarray(X, dtype=np.float32)) / 255.0
        data_loader=DataLoader(TensorDataset(X_data), batch_size=self.batch_size,
                            shuffle=False, num_workers=0)
        y_pred=[]
        with torch.no_grad():
            for(X_batch,) in data_loader:
                output=self.forward(X_batch.to(self.device))
                y_pred.append(output.argmax(dim=1).cpu().numpy())
        return np.concatenate(y_pred)

    def run(self):
        print('method running...')
        print('--start training...')
        self.fit(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y=self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
