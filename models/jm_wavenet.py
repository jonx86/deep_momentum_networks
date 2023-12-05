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
from collections import OrderedDict
from torch.optim.lr_scheduler import ReduceLROnPlateau

############## SET SEED ############## 
torch.manual_seed(0)
######################################

class CausalConv1d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, dilation, device):
		super(CausalConv1d, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.padding = dilation*(kernel_size-1)
		self.conv = nn.Conv1d(in_channels, out_channels,
 				              kernel_size, padding=self.padding,
 				              dilation=dilation, device=device)

	def forward(self, inputs):
		outputs = self.conv(inputs)
		return outputs[:, :, :-self.padding]
      

class ResBlock(nn.Module):
      def __init__(self, in_channels, out_channels, kernel_size, dilation, device):
            super(ResBlock, self).__init__()
            self.in_channels=in_channels
            self.out_channels=out_channels
            self.kernel_size=kernel_size
            self.dilation = dilation

            self.DilatedConv = CausalConv1d(in_channels=self.in_channels,
                                            out_channels=self.out_channels,
                                            kernel_size=self.kernel_size,
                                            dilation=self.dilation,
                                            device=device)
            
            self.residConnection = nn.Conv1d(in_channels=self.in_channels,
                                             out_channels=self.out_channels,
                                             kernel_size=1, dilation=1,
                                             device=device)
            
            self.skipConnection = nn.Conv1d(in_channels=self.in_channels,
                                            out_channels=self.out_channels,
                                            kernel_size=1, dilation=1,
                                            device=device)
            
            self.tanh = nn.Tanh()
            self.sigmoid = nn.Sigmoid()
            

      def forward(self, inputs):
            # outputs
            outputs = self.DilatedConv(inputs)
            tanh, sig = self.tanh(outputs), self.sigmoid(outputs)
            outputs = tanh * sig

            # resid connection 
            resid_connection = self.residConnection(outputs) + inputs

            # skip connection
            skip_connection = self.skipConnection(outputs)
            return resid_connection, skip_connection
      

class ResStack(nn.Module):
      def __init__(self, layers:list, stacks, in_channels, out_channels, kernel_size, device):
            super(ResStack, self).__init__()
            self.layers = layers
            self.stacks = stacks
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size

            # save all the stacks
            self.stack_list = []

            # loop through stacks and layers
            for _ in range(self.stacks):
                  single_stack = []
                  for dilation in layers:
                        r = ResBlock(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=kernel_size,
                                     dilation=dilation, device=device)
                        single_stack.append(r)
                  self.stack_list.append(single_stack)
      
      def forward(self, inputs):
            skips = []
            residual= inputs
            for stack in self.stack_list:
                  for layer in stack:
                        residual, skip = layer(residual)
                        skips.append(skip)

            # skips?
            skips = torch.stack(skips)
            return skips


class FullyConnectedBlock(nn.Module):
      def __init__(self, hidden, out_dim=1, dropout_rate=.30):
            super(FullyConnectedBlock, self).__init__()
            self.hidden = hidden
            self.out_dim=out_dim
            self.dropout_rate=dropout_rate

            self.seq = nn.Sequential(
                  nn.LazyLinear(self.hidden),
                  nn.Tanh(),
                  nn.Dropout(p=self.dropout_rate),
                  nn.LazyLinear(out_dim),
                  nn.Tanh())
            
      def forward(self, inputs):
            outputs = self.seq(inputs)
            return outputs
      

class FullyConnectedBlock1L(nn.Module):
      def __init__(self, out_dim=1):
            super(FullyConnectedBlock1L, self).__init__()
            self.out_dim=out_dim
            self.seq = nn.Sequential(
                  nn.LazyLinear(self.out_dim),
                  nn.Tanh())
            
      def forward(self, inputs):
            outputs = self.seq(inputs)
            return outputs
      

class OnebyOneBlock(nn.Module):
       def __init__(self, kernel, in_channels, out_channels, out_dim=1):
            super(OnebyOneBlock, self).__init__()
            self.kernel=kernel
            self.in_channels=in_channels
            self.out_channels=out_channels
            self.out_dim=out_dim

            self.seq = nn.Sequential(
                   nn.Conv1d(in_channels=self.in_channels,
                             out_channels=self.out_channels,
                             kernel_size=1),
                   nn.Tanh(),
                   nn.Conv1d(in_channels=self.out_channels,
                             out_channels=out_dim,
                             kernel_size=1),
                   nn.Tanh())
      
       def forward(self, inputs):
              outputs = self.seq(inputs)
              return outputs.squeeze(1)[:, -1]
       

class WaveNet(nn.Module):

      def __init__(self, in_channels, out_channels, kernal_size, num_layers, num_stacks, hidden_size, dropOutRate=.30, device='cuda'):
            super(WaveNet, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernal_size = kernal_size
            self.num_layers = num_layers
            self.num_stacks = num_stacks
            self.hidden_size = hidden_size
            self.dropOutRate = dropOutRate
            self.device = device


            self.CausalConvFirst = CausalConv1d(in_channels=self.in_channels,
                                                out_channels=self.out_channels,
                                                kernel_size=self.kernal_size,
                                                dilation=1, device=self.device)
            
            self.ResStacks = ResStack(layers=self.num_layers,
                                      stacks=self.num_stacks,
                                      in_channels=self.out_channels,
                                      kernel_size=self.kernal_size,
                                      out_channels=self.out_channels,
                                      device=self.device)
            
            self.tanh1 = nn.Tanh()
           
            # drop out
            self.drop1 = nn.Dropout(p=self.dropOutRate)
            self.flattener = nn.Flatten()

            # last we have our full connected block
            self.fcBlock = FullyConnectedBlock(hidden=self.hidden_size,
                                               out_dim=1,
                                               dropout_rate=self.dropOutRate)
            

      def forward(self, inputs):
            # pass through the first casual convolutional layer
            outputs = self.CausalConvFirst(inputs)

            # now we go into the residual block
            skip_connections = self.ResStacks(outputs)
            skip_connections = torch.sum(skip_connections, dim=0)
            skip_connections = self.tanh1(skip_connections)

            # flatten before last fully connected layer
            skip_connections = self.flattener(skip_connections)
            skip_connections = self.fcBlock(skip_connections)
            return skip_connections
            

# bring in the data
data = load_features()

# create the additional lags shown in the paper
features = [f for f in data.columns if f.startswith('feature')]
lags = [l for l in data.columns if l.startswith('lag')]
target = ['target']
both = features+target+lags

full = data[both].dropna(subset=both)
print(full.shape)
X = full[both]
features+=lags


# NOTE - Please tune, batch size, learning rate, hidden size, dropout, max grad norm
model_path = 'model_WaveNet.pt'
MODEL_NAME = 'WaveNet'
IN_CHANNELS = 60
HIDDEN_DIM = 40
OUT_CHANNELS = 3
SEC_LEN=63
DROPOUT_RATE = .50
BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = 'cuda'
NUM_CORES = 4 # -1 for all cores, there are 3 multi-processed data aggregate functions, because we need to operate on the future level
EARLY_STOPPING = 25
KERNEL_SIZE = 2

prep = PrePTestSeqData(X)

# TODO split Xy for seq could to be multi-processed for now this is slow but works - Use joblib
# My method to process the entire dataset first so I can filter on correct dates for test set and
# don't need to look back a small window into train set

FILENAME = f"xs_{SEC_LEN}_{MODEL_NAME}.pickle" # incase we want to test different sequence lenghts

try:
      with open(FILENAME, "rb") as f:
                  print('loading pickle .....')
                  start = time.time()
                  newX = pickle.load(f)
                  print(f"Loading pickle took: {time.time() - start}")
except Exception:
      newX, _ = split_Xy_for_seq(X[features], X['target'],
                                  step_size=SEC_LEN,
                                  return_pandas=True,
                                  return_seq_target=False,
                                  lstm=False)

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
                              lstm=False,
                              return_seq_target=False)
      
       X_train2, y_train2 = split_Xy_for_seq(X_train=X_train2,
                                    y_train=y_train2,
                                    step_size=SEC_LEN,
                                    return_pandas=False,
                                    lstm=False,
                                    return_seq_target=False)
      
       train_loader = load_data_torch(X_train2, y_train2,
                                    batch_size=BATCH_SIZE,
                                    device=DEVICE)
      
       val_loader = load_data_torch(X_val, y_val,
                                    batch_size=BATCH_SIZE,
                                    device=DEVICE)
       
       dilations = [5, 10, 15, 21, 42]
       model = WaveNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, kernal_size=KERNEL_SIZE,
                       num_layers=dilations, num_stacks=1, hidden_size=HIDDEN_DIM,
                       dropOutRate=DROPOUT_RATE)
       
       model.to(torch.device(DEVICE))
       optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
       scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=.5, verbose=True)
       loss_fnc = SharpeLoss(risk_trgt=.15)

       early_stop_count = 0
       best_val_loss = float('inf')
       for epoch in range(EPOCHS):
             train_loss = train_model(epoch,
                                      model,
                                      train_loader=train_loader,
                                      optimizer=optimizer,
                                      loss_fnc=loss_fnc,
                                      clip_norm=True,
                                      max_norm=.001,
                                      device=DEVICE)
             
             val_loss = validate_model(epoch, model, val_loader,
                                       loss_fnc=loss_fnc,
                                       device=DEVICE)
             
             scheduler.step(val_loss)
            
             # append learning curves
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
       plt.savefig(f'learning_curves_{idx}_{MODEL_NAME}.png')
       plt.clf()

       learning_curves.loc[epoch, 'train_loss'] = train_loss
       learning_curves.loc[epoch, 'val_loss'] = val_loss

       # we need a tuple of the test set start and end dates
       test_start, test_end = X_test.index.get_level_values('date')[0], X_test.index.get_level_values('date')[-1]
       print(f'Test Start :{test_start} | Test End :{test_end}')

       # we don't need X train anymore
       del X_train

       xs1 = mpSplits(prep.split_single_future,
                     (test_start, test_end), newX,
                     n_jobs=NUM_CORES)
       
       model.load_state_dict(torch.load(model_path))
       with torch.no_grad():
                model.eval()
                # feed in sequences for each future and get the predictions, take just the last time-step
                preds = aggregate_seq_preds(model, xs1,
                                            features=features,
                                            device=DEVICE,
                                            lstm=False,
                                            seq_out=False,
                                            n_jobs=NUM_CORES)
                
                preds = preds.to_frame('WaveNet')
                predictions.append(preds)


preds = pd.concat(predictions).sort_index()
feats = data.join(preds['WaveNet'], how='left')
feats.dropna(subset=['WaveNet'], inplace=True)
dates = feats.index.get_level_values('date').unique().to_list()
strat_rets = process_jobs(dates, feats, signal_col='WaveNet')
bt = get_returns_breakout(strat_rets.fillna(0.0).to_frame('WaveNet'))
print(bt)




