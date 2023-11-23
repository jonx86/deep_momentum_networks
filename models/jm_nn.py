import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import time

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from utils.utils import (get_cv_splits,
                         load_features,
                         train_val_split,
                         process_jobs,
                         get_returns_breakout)

from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from losses.jm_loss import SharpeLoss

class MLP(nn.Module):
    def __init__(self, hidden_size=5, dropout=.30, input_dim=9, output_dim=1):
        super(MLP, self).__init__()

        # init the params
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim

        # init the layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_size)
        self.relU1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relU2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_size, self.output_dim)

        # dorp out on the first two layers only?
        self.dropOut1 = nn.Dropout(p=self.dropout)
        self.dropOut2 = nn.Dropout(p=self.dropout)

        # pred activation function
        self.fc3act = nn.Tanh()

    def forward(self, inputs):
        # roll through model
        inputs = self.relU1(self.dropOut1(self.fc1(inputs)))
        inputs = self.relU2(self.dropOut2(self.fc2(inputs)))
        inputs = self.fc3(inputs)

        # now for the predictions
        inputs = self.fc3act(inputs)
        return inputs


if __name__ == '__main__':
    feats = load_features()

    # grab the model features and targets
    features = [f for f in feats.columns if f.startswith('feature')]
    target = ['fwd_ret1d', 'rVol']
    full = features + target

    # condense and group
    X = feats[full].dropna()
    X.dropna(inplace=True)
   
    # returns the loader
    def load_data_torch(X, y, batch_size=64):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)

        # send to cuda
        X.to(torch.device('cuda'))
        y.to(torch.device('cuda'))

        loader = DataLoader(list(zip(X, y)), shuffle=False, batch_size=batch_size)
        return loader
    

    def validate_model(epoch, model, val_loader, loss_fnc):
        iter_time = AverageMeter()
        losses = AverageMeter()

        for idx, (data, target) in enumerate(val_loader):
            start = time.time()

            if torch.cuda.is_available:
                data = data.cuda()
                target = target.cuda()
            
            with torch.no_grad():
                out = model(data)
                out = out.cuda()
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


    def train_model(epoch, model, train_loader, optimizer, loss_fnc):
        iter_time = AverageMeter()
        losses = AverageMeter()
      
        for idx, (data, target) in enumerate(train_loader):
            start = time.time()

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            # forward step
            out = model(data)
            out = out.cuda()
            loss = loss_fnc(out, target)

            # gradient descent step
            optimizer.zero_grad()
            loss.backward()
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

    
    # params
    EPOCHS = 100
    learning_rate = 1e-4
    batch_size = 256
    hidden_layer_size = 80
    dropout_rate = .50
    #reg = 1e-5

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


    predictions = []

    # now start the loop
    for idx, (train, test) in enumerate(get_cv_splits(X)):
         learning_curves = pd.DataFrame(columns=['train_loss', 'val_loss'])
         iter_time = AverageMeter()
         train_losses = AverageMeter()
         
         # break out X and y train
         X_train, y_train = train[features], train[target] 
         X_test, y_test = test[features], test[target]

         # validation split
         X_train2, X_val, y_train2, y_val = train_val_split(X_train, y_train)

         scaler = RobustScaler()
         X_train = scaler.fit_transform(X_train)

         # now scaler X_train2 and Xval
         X_train2 = scaler.transform(X_train2)
         X_val = scaler.transform(X_val)

         # our data-loaders
         dataloader = load_data_torch(X_train2, y_train2, batch_size=batch_size)
         valdataloader = load_data_torch(X_val, y_val, batch_size=batch_size)

         # model
         model = MLP(hidden_size=hidden_layer_size, dropout=dropout_rate)
         model.to(torch.device('cuda'))

         optimizer = Adam(model.parameters(), lr=learning_rate)
         loss_func = SharpeLoss(risk_trgt=.15)

         for epoch in range(EPOCHS):
             train_loss = train_model(epoch, model, dataloader, optimizer, loss_func)
             val_loss = validate_model(epoch, model, valdataloader, loss_func)

             learning_curves.loc[epoch, 'train_loss'] = train_loss
             learning_curves.loc[epoch, 'val_loss'] = val_loss

         learning_curves.plot()
         plt.title(f'CV Split: {idx}')
         plt.savefig(f'learning_curves_{idx}.png')
         plt.clf()

         # scale 
         X_test2 = X_test.copy()
         X_test2 = scaler.transform(X_test2)
        
         X_test2 = torch.tensor(X_test2, dtype=torch.float32)
         X_test2 = X_test2.cuda()

         with torch.no_grad():
            model.eval()
            preds = model(X_test2)
            preds = preds.cpu().detach().numpy()
            preds=preds.reshape(preds.shape[0], )
            preds = pd.Series(data=preds, index=y_test.index)
            predictions.append(preds)

    preds=pd.concat(predictions).sort_index()
    preds = preds.to_frame('mlp')
    feats = feats.join(preds[['mlp']], how='left')
    feats.dropna(subset=['mlp'], inplace=True)
    dates = feats.index.get_level_values('date').unique().to_list()
    strat_rets = process_jobs(dates, feats, signal_col='mlp')
    print(get_returns_breakout(strat_rets.fillna(0.0).to_frame('mlp_bench')))


       

            
            



             







