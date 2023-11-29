import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import ParameterSampler
import time

from sklearn.preprocessing import RobustScaler

from utils.utils import *

from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from losses.jm_loss import SharpeLoss, RetLoss


data = load_features()
features = [f for f in data.columns if f.startswith('feature')]
target = ['target']
both = features+target

full = data[both].dropna(subset=both)
print(full.shape)

X = full[both]


class simpleLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, dropout_rate=.30):
                super(simpleLSTM, self).__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.dropout_rate = dropout_rate

                self.lstm = nn.LSTM(input_size=self.input_dim,
                                    hidden_size=self.hidden_dim,
                                    num_layers=1,
                                    batch_first=True)
                self.lstm.flatten_parameters()
                
                self.linear = nn.Linear(self.hidden_dim, 1)
                self.dropOut1 = nn.Dropout()

        def forward(self, inputs):
                #https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506"
                outputs, _ = self.lstm(inputs)
                outputs = torch.tanh(self.linear(outputs))
                # we want to return the prediction at the last time-step , this is the position size at t+1
                return outputs[:, -1, :]

HIDDEN_DIM = 20
INPUT = 20
DROPOUT_RATE = .30
BATCH_SIZE = 256
EPOCHS = 25
LEARNING_RATE = 1e-3
DEVICE = 'cpu'

# used for getting the correct batching of the test set to feed into LSTM
prep = PrePTestSeqData(X)
newX, _ = split_Xy_for_seq(X[features], X['target'], step_size=INPUT, lstm=True)

for idx, (train, test) in enumerate(get_cv_splits(X)):
        learning_curves = pd.DataFrame(columns=['train_loss', 'val_loss'])
        iter_time = AverageMeter()
        train_losses = AverageMeter()
        
        # break out X and y train
        X_train, y_train = train[features], train[target] 
        X_test, y_test = test[features], test[target]

        # validation split
        X_train2, X_val, y_train2, y_val = train_val_split(X_train, y_train)

        # scale the data
        scaler = RobustScaler()
        scaler.fit(X_train2)

        # We want to fit only on the 90% xVal
        X_train2 = retain_pandas_after_scale(X_train2, scaler=scaler)
        X_val = retain_pandas_after_scale(X_val, scaler=scaler)

        # this function 
        X_val, _ = split_Xy_for_seq(X_train=X_val,
                                    y_train=y_val,
                                    step_size=INPUT,
                                    return_pandas=False,
                                    lstm=True)
        
        X_train2, _ = split_Xy_for_seq(X_train=X_train2,
                                       y_train=y_train2,
                                       step_size=INPUT,
                                       return_pandas=False,
                                       lstm=True)
        
        train_loader = load_data_torch(X_train2, y_train2,
                                       batch_size=BATCH_SIZE,
                                       device=DEVICE)
        
        val_loader = load_data_torch(X_val, y_val,
                                     batch_size=BATCH_SIZE,
                                     device=DEVICE)

        model = simpleLSTM(input_dim=INPUT,
                           hidden_dim=HIDDEN_DIM,
                           dropout_rate=DROPOUT_RATE)
        
        model.to(torch.device(DEVICE))
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fnc = SharpeLoss(risk_trgt=.15)
        
        # now iterate though all the epochs
        for epoch in range(EPOCHS):
                train_loss = train_model(epoch, 
                                         model,
                                         train_loader=train_loader,
                                         optimizer=optimizer,
                                         loss_fnc=loss_fnc,
                                         clip_norm=True,
                                         max_norm=.001,
                                         device=DEVICE)
                
                val_loss = validate_model(epoch,
                                          model,
                                          val_loader,
                                          loss_fnc=loss_fnc,
                                          device=DEVICE)
                
        X_test2 = X_test.copy()
        X_test2 = scaler.transform(X_test2)

        X_test2 = X_test.copy()
        X_test2 = retain_pandas_after_scale(X_test2, scaler)

        # we need a tuple of the test set start and end dates
        test_start, test_end = X_test.index.get_level_values('date')[0], X_test.index.get_level_values('date')[-1]
        print(f'Test Start :{test_start} | Test End :{test_end}')


        #NOTE get correct test data takes a long time looping through all 72
        # futures and all over-lapping sequences with in each future
        xs1 = prep.run_all_splits((test_start, test_end), newX)

        with torch.no_grad():
                model.eval()
                preds = aggregate_seq_preds(model, xs1, features=features, device=DEVICE)
        break

                

        
                
        

        
        

