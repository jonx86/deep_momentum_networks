import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import RobustScaler

import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from losses.ryan_losses import SharpeLoss
from models.ryan_lstm import LSTM
from utils.utils import *


# Dataset
dataset = load_features()

# Inputs and Targets
features = [column for column in dataset.columns if column.startswith('feature')]
target = ['target']

# Workingset
X = dataset[features + target].dropna()

# Parameters
model_path = 'model.pt'
early_stopping = 25
device = torch.device('cuda')
number_of_features = 10
epochs = 100
dropout = 0.3
hidden_size = 20
batch_size = 2048
learning_rate = 0.001
maximum_gradient_norm = 0.01

# used for getting the correct batching of the test set to feed into LSTM
prep = PrePTestSeqData(X)
newX, _ = split_Xy_for_seq(X[features], X['target'], step_size=63, lstm=True)

for idx, (train, test) in enumerate(get_cv_splits(X)):
        if idx > 2:
          break

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
                                    step_size=63,
                                    return_pandas=False,
                                    lstm=True)

        X_train2, _ = split_Xy_for_seq(X_train=X_train2,
                                       y_train=y_train2,
                                       step_size=63,
                                       return_pandas=False,
                                       lstm=True)

        train_loader = load_data_torch(X_train2, y_train2,
                                       batch_size=batch_size,
                                       device=device)

        val_loader = load_data_torch(X_val, y_val,
                                     batch_size=batch_size,
                                     device=device)

        model = LSTM(number_of_features, hidden_size, dropout=dropout, device=device)

        model.to(torch.device(device))
        optimizer = Adam(model.parameters(), lr=learning_rate)
        loss_fnc = SharpeLoss(risk_trgt=.15)

        # now iterate though all the epochs
        for epoch in range(epochs):
                train_loss = train_model(epoch,
                                         model,
                                         train_loader=train_loader,
                                         optimizer=optimizer,
                                         loss_fnc=loss_fnc,
                                         clip_norm=True,
                                         max_norm=maximum_gradient_norm,
                                         device=device)

                val_loss = validate_model(epoch,
                                          model,
                                          val_loader,
                                          loss_fnc=loss_fnc,
                                          device=device)

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

        model.eval()
        with torch.no_grad():
                preds = aggregate_seq_preds(model, xs1, features=features, device=device)
        break
