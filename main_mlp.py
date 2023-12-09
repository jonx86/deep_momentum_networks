import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import RobustScaler

import time

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
#from models.jm_nn import MLP
from losses.jm_loss import SharpeLoss
from utils.utils import (load_features, get_cv_splits,
                         train_val_split, process_jobs,
                         get_returns_breakout,
                        MLP_FEATURES)

import os
torch.manual_seed(0)
from torch.optim.lr_scheduler import ReduceLROnPlateau

outfile = open("outputs", "w")


class MLP(nn.Module):
    def __init__(self, hidden_size=5, dropout=.30, input_dim=60, output_dim=1):
        super(MLP, self).__init__()

        # init the params
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim

        # init the layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_dim)

        # dorp out on the first two layers only?
        self.dropOut1 = nn.Dropout(p=self.dropout)
        self.dropOut2 = nn.Dropout(p=self.dropout)
    
    def forward(self, inputs):
        # roll through model
        outputs = self.dropOut1(torch.tanh(self.fc1(inputs)))
        outputs = self.dropOut2(torch.tanh(self.fc2(outputs)))
        #print('Size of Outputs', outputs.shape)
        return outputs


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_data_torch(X, y, batch_size=64, device='cpu'):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32)

    X = X.to(torch.device(device=device))
    y = y.to(torch.device(device=device))

    loader = DataLoader(list(zip(X, y)), shuffle=False, batch_size=batch_size)
    return loader


def validate_model(epoch, model, val_loader, loss_fnc):
    iter_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()

        with torch.no_grad():
            out = model(data)
            loss = loss_fnc(out, target)

        losses.update(loss.item(), out.shape[0])
        iter_time.update(time.time() - start)

        if idx % 10==0:
                print(('Epoch: [{0}][{1}/{2}]\t'
                'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t').format(
                    epoch,
                    idx,
                    len(val_loader),
                    iter_time=iter_time,
                    loss=losses))

    return losses.avg


def train_model(epoch, model, train_loader, optimizer, loss_fnc, clip):
    iter_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    for idx, (data, target) in enumerate(train_loader):
        start = time.time()

        # forward step
        out = model(data)
        loss = loss_fnc(out, target)

        # gradient descent step
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        losses.update(loss.item(), out.shape[0])
        iter_time.update(time.time() - start)

        if idx % 10 == 0:
                print(('Epoch: [{0}][{1}/{2}]\t'
                'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t').format(
                    epoch,
                    idx,
                    len(train_loader),
                    iter_time=iter_time,
                    loss=losses))
    return losses.avg


# Dataset
dataset = load_features()
dataset.dropna(inplace=True)

# Future IDs
futures = dataset.index.get_level_values('future').unique().tolist()
target = ['target']

# Workingset
X = dataset[MLP_FEATURES + target]

# Parameters
model_path = 'model.pt'
early_stopping = 25
device = torch.device('cuda')
number_of_features = 8
epochs = 100
dropout = 0.2
hidden_size = 20
batch_size = 512
learning_rate = 1e-3
maximum_gradient_norm = 0.01
lr_scheduler = True
val_pct = .80

if os.path.exists(model_path):
    os.remove(model_path)

predictions = [0] * 6
# now start the loop
for idx, (train, test) in enumerate(get_cv_splits(X)):
    learning_curves = pd.DataFrame(columns=['train_loss', 'val_loss'])
    iter_time = AverageMeter()
    train_losses = AverageMeter()

    # break out X and y train
    X_train, y_train = train[MLP_FEATURES], train[target]
    X_test, y_test = test[MLP_FEATURES], test[target]

    # validation split
    X_train2, X_val, y_train2, y_val = train_val_split(X_train, y_train, train_pct=val_pct)

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)

    # now scaler X_train2 and Xval
    X_train2 = scaler.transform(X_train2)
    X_val = scaler.transform(X_val)

    # our data-loaders
    dataloader = load_data_torch(X_train2, y_train2, batch_size=batch_size, device=device)
    valdataloader = load_data_torch(X_val, y_val, batch_size=batch_size, device=device)

    # model
    model = MLP(hidden_size=hidden_size, dropout=dropout, input_dim=48)
    model.to(torch.device('cuda'))
    if os.path.exists(model_path):
        print('model loaded')
        model.load_state_dict(torch.load(model_path))
        os.remove(model_path)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=.5, verbose=True)
    loss_func = SharpeLoss()

    early_stop_count = 0
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train_model(epoch, model, dataloader,
                                 optimizer, loss_func,
                                 maximum_gradient_norm)
        
        val_loss = validate_model(epoch, model,
                                  valdataloader,
                                  loss_func)

        if lr_scheduler:
            scheduler.step(val_loss)

        learning_curves.loc[epoch, 'train_loss'] = train_loss
        learning_curves.loc[epoch, 'val_loss'] = val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count == early_stopping:
            break

    learning_curves.plot()
    plt.title(f'CV Split: {idx}')
    plt.savefig(f'learning_curves_{idx}.png')
    plt.clf()

    # scale
    X_test2 = X_test.copy()
    X_test2 = scaler.transform(X_test2)
    X_test2 = torch.tensor(X_test2, dtype=torch.float32)
    X_test2 = X_test2.to(device)

    model.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        model.eval()
        preds = model(X_test2)
        preds = preds.cpu().detach().numpy()
        preds=preds.reshape(preds.shape[0], )
        preds = pd.Series(data=preds, index=y_test.index)
        predictions[idx] = preds

    preds=pd.concat([predictions[idx]]).sort_index()
    preds = preds.to_frame('mlp')
    new_dataset = dataset.copy()
    feats = new_dataset.join(preds[['mlp']], how='left')
    feats.dropna(subset=['mlp'], inplace=True)
    dates = feats.index.get_level_values('date').unique().to_list()
    strat_rets = process_jobs(dates, feats, signal_col='mlp')
    values = get_returns_breakout(strat_rets.fillna(0.0).to_frame('mlp_bench'))
    print('idx: ', idx, file=outfile)
    print(values, file=outfile, flush=True)
    torch.save(model.state_dict(), model_path)
    break
   

# preds=pd.concat(predictions).sort_index()
# preds = preds.to_frame('mlp')
# feats = dataset.join(preds[['mlp']], how='left')
# feats.dropna(subset=['mlp'], inplace=True)
# dates = feats.index.get_level_values('date').unique().to_list()
# strat_rets = process_jobs(dates, feats, signal_col='mlp')
# print('Final', file=outfile, flush=True)
# print(get_returns_breakout(strat_rets.fillna(0.0).to_frame('mlp_bench')), file=outfile, flush=True)
# outfile.close()

   

