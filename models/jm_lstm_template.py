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
from models.ryan_transformer import Transformer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from sklearn.model_selection import ParameterSampler

from models.ryan_transformer_encoder import TransformerEncoder

############## SET SEED ############## 
torch.manual_seed(0)
######################################

data = load_features()
features = [f for f in data.columns if f.startswith('feature')]
target = ['target']
both = features+target

full = data[both].dropna(subset=both)
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
                
                # you may or may not need to do this, LSTM does not work on GPU for me 
                # and can't fix the bug as of now, params needs to be same memory block 
                #https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506"
                self.lstm.flatten_parameters()
                
                self.linear = nn.Linear(self.hidden_dim, 1)
                self.dropOut1 = nn.Dropout()

        def forward(self, inputs):
                outputs, _ = self.lstm(inputs)
                outputs = torch.tanh(self.linear(outputs))
                # we want to return the prediction at the last time-step , this is the position size at t+1
                return outputs
        

class FullLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, dropout_rate=.30):
                super(FullLSTM, self).__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.dropout_rate = dropout_rate

                self.lstm = nn.LSTM(input_size=self.input_dim,
                                    hidden_size=self.hidden_dim,
                                    num_layers=1,
                                    batch_first=True)
                
                # you may or may not need to do this, LSTM does not work on GPU for me 
                # and can't fix the bug as of now, params needs to be same memory block 
                #https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506"
                self.lstm.flatten_parameters()
                
                # fully connecte4d block, 2-layer MLP
                self.fcBlock = nn.Sequential(
                        nn.LazyLinear(self.hidden_dim),
                        nn.Tanh(),
                        nn.Dropout(p=self.dropout_rate),
                        nn.LazyLinear(out_features=1),
                        nn.Tanh())

        def forward(self, inputs):
                outputs, _ = self.lstm(inputs)
                outputs = self.fcBlock(outputs)
                return outputs


# early stopping
EARLY_STOPPING = 25
model_path = 'transformer_encode.pt'
HIDDEN_DIM = 40
INPUT = 10
SEC_LEN=63
DROPOUT_RATE = .30
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = 'cpu'
NUM_CORES = 8 # -1 for all cores, there are 3 multi-processed data aggregate functions, because we need to operate on the future level

# used for getting the correct batching of the test set to feed into LSTM
prep = PrePTestSeqData(X)

# TODO split Xy for seq could to be multi-processed for now this is slow but works - Use joblib
# My method to process the entire dataset first so I can filter on correct dates for test set and
# don't need to look back a small window into train set

FILENAME = f"xs_{SEC_LEN}.pickle" # incase we want to test different sequence lenghts

try:
    with open(FILENAME, "rb") as f:
            print('loading pickle .....')
            start = time.time()
            newX = pickle.load(f)
            print(f"Loading pickle took: {time.time() - start}")
except Exception:
        newX, _ = split_Xy_for_seq(X[features], X['target'],
                                   step_size=SEC_LEN,
                                   return_seq_target=True,
                                   lstm=True)

        with open(FILENAME, "wb") as f:
                pickle.dump(newX, f)
                print("dumped file")
        

predictions = []
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
                                    step_size=SEC_LEN,
                                    return_pandas=False,
                                    lstm=True,
                                    return_seq_target=True)
        
        X_train2, y_train2 = split_Xy_for_seq(X_train=X_train2,
                                       y_train=y_train2,
                                       step_size=SEC_LEN,
                                       return_pandas=False,
                                       lstm=True,
                                       return_seq_target=True)
        
        train_loader = load_data_torch(X_train2, y_train2,
                                       batch_size=BATCH_SIZE,
                                       device=DEVICE)
        
        val_loader = load_data_torch(X_val, y_val,
                                     batch_size=BATCH_SIZE,
                                     device=DEVICE)

        model = simpleLSTM(input_dim=INPUT, hidden_dim=HIDDEN_DIM)
        
        model.to(torch.device(DEVICE))
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=.5, verbose=True)
        loss_fnc = SharpeLoss(risk_trgt=.15)
        
        early_stop_count = 0
        best_val_loss = float('inf')
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
                
                scheduler.step(val_loss)
                learning_curves.loc[epoch, 'train_loss'] = train_loss
                learning_curves.loc[epoch, 'val_loss'] = val_loss
      
                if val_loss<best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), model_path)
                        early_stop_count += 0
                else:
                        early_stop_count +=1

                if early_stop_count == EARLY_STOPPING:
                        print(f'Early Stopping Applied on Epoch: {epoch}')
                        break

        learning_curves.plot()
        plt.title(f'CV Split: {idx}')
        plt.savefig(f'learning_curves_{idx}_LSTM.png')
        plt.clf()
                
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

        model.load_state_dict(torch.load(model_path))
        with torch.no_grad():
                model.eval()
                # feed in sequences for each future and get the predictions, take just the last time-step
                preds = aggregate_seq_preds(model, xs1,
                                            features=features,
                                            device=DEVICE,
                                            lstm=True,
                                            seq_out=True,
                                            n_jobs=NUM_CORES)
                
                preds = preds.to_frame('lstm')
                predictions.append(preds)

        # run back-test
        preds = pd.concat(predictions).sort_index()
        feats = data.join(preds['lstm'], how='left')
        feats.dropna(subset=['lstm'], inplace=True)
        dates = feats.index.get_level_values('date').unique().to_list()
        strat_rets = process_jobs(dates, feats, signal_col='lstm')
        bt = get_returns_breakout(strat_rets.fillna(0.0).to_frame('lstm_test'))
        print(bt)
        break

      
                

        
                
        

        
        

