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

























HIDDEN_DIM = 20
INPUT = 63
DROPOUT_RATE = .30
BATCH_SIZE = 256
EPOCHS = 25
LEARNING_RATE = 1e-3
DEVICE = device
NUM_CORES = -1 # -1 for all cores, there are 3 multi-processed data aggregate functions, because we need to operate on the future level

print("AAA")


# used for getting the correct batching of the test set to feed into LSTM
prep = PrePTestSeqData(X)


print("BBB")

# TODO split Xy for seq could to be multi-processed for now this is slow but works - Use joblib
# My method to process the entire dataset first so I can filter on correct dates for test set and
# don't need to look back a small window into train set
newX, _ = split_Xy_for_seq(X[features], X['target'], step_size=INPUT, lstm=True)


print("CCC")

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
        X_val, y_val = split_Xy_for_seq(X_train=X_val,
                                    y_train=y_val,
                                    step_size=INPUT,
                                    return_pandas=False,
                                    lstm=True,
                                    return_seq_target=True)

        X_train2, y_train2 = split_Xy_for_seq(X_train=X_train2,
                                       y_train=y_train2,
                                       step_size=INPUT,
                                       return_pandas=False,
                                       lstm=True,
                                       return_seq_target=True)

        train_loader = load_data_torch(X_train2, y_train2,
                                       batch_size=BATCH_SIZE,
                                       device=DEVICE)

        val_loader = load_data_torch(X_val, y_val,
                                     batch_size=BATCH_SIZE,
                                     device=DEVICE)

        model = LSTM(10, HIDDEN_DIM, dropout=DROPOUT_RATE, device=device)

        model.to(torch.device(DEVICE))
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fnc = SharpeLoss()

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


        #NOTE get correct test data takes a long time looping through all 72 futures
        # futures and all over-lapping sequences with in each future
        # We utilize joblib again to multi-process these batches, looping through sequences within each future
        # takes ~1.6 minutes on my machine 24-cores
        xs1 = mpSplits(prep.split_single_future,
                       (test_start, test_end),
                        newX, n_jobs=NUM_CORES)

        with torch.no_grad():
                model.eval()
                # feed in sequences for each future and get the predictions, take just the last time-step
                preds = aggregate_seq_preds(model, xs1, features=features,
                                            device=DEVICE, lstm=True,
                                            n_jobs=NUM_CORES)

                preds = preds.to_frame('lstm')
                feats = data.join(preds['lstm'], how='left')
                feats.dropna(subset=['lstm'], inplace=True)
                dates = feats.index.get_level_values('date').unique().to_list()
                strat_rets = process_jobs(dates, feats, signal_col='lstm')
                bt = get_returns_breakout(strat_rets.fillna(0.0).to_frame('lstm_test'))
                print(bt)
        break
