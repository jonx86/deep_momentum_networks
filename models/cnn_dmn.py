import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import ParameterSampler
import time

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from utils.utils import (get_cv_splits,
                         load_features,
                         train_val_split,
                         process_jobs,
                         get_returns_breakout,
                         split_sequence_for_cnn,
                         split_Xy_for_seq,
                         retain_pandas_after_scale,
                         split_rolling_sequences_for_cnn,
                         aggregate_seq_preds,
                         PrePTestSeqData)

from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from losses.jm_loss import SharpeLoss, RetLoss


class SimpleCNN(nn.Module):

    def __init__(self, in_channels, hidden_shape, output_shape, filter_size=64, kernel_size=2, droput_rate=.30,
                 pool_size=2):
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.dropout_rate = droput_rate
        self.pool_size=pool_size


        # 1d convolutional layer
        self.conv1d = nn.Conv1d(in_channels=self.in_channels,
                                out_channels=self.filter_size,
                                kernel_size=self.kernel_size)
        
        self.maxPool = nn.AvgPool1d(kernel_size=self.pool_size)
        self.flatter = nn.Flatten()

        # now two fully connected layers
        self.fc1 = nn.LazyLinear(out_features=self.hidden_shape)
        self.fc2 = nn.LazyLinear(out_features=self.output_shape)

        # dropout1
        self.dropOut1 = nn.Dropout(p=self.dropout_rate)
        self.dropOut2 = nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs):
        outputs = self.conv1d(inputs)
        outputs = self.maxPool(outputs)

        #new shape should be N, Filter * Seq Length - Kernel + 1
        outputs = self.flatter(outputs)
        outputs = self.dropOut1(torch.tanh(self.fc1(outputs)))
        outputs = self.dropOut2(torch.tanh(self.fc2(outputs)))

        return outputs
    

class SimpleLSTM(nn.Module):

    def __init__(self, input_dim, sequence_length, hidden_size, dropout_rate=.30, lstm_layers=2):
        super(SimpleLSTM, self).__init__()

        self.input_dim = input_dim
        self.sequence_length=sequence_length
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.lstm_layers = 2

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.sequence_length,
                            dropout=self.dropout_rate,
                            num_layers=2)
        
        # all the layers
        self.flatter = nn.Flatten()
        self.fc1 = nn.LazyLinear()
        self.fc2 = nn.LazyLinear()

        # drop outs
        self.dropout1 = nn.Dropout(p=self.dropout_rate)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)
        
    def forward(inputs, h, ct):
        pass
        
        
if __name__ == '__main__':
    sampleX = np.random.randn(100, 60, 1) # where we have batch size, feature dim, temporal dim
    sampleX_second = np.random.randn(20, 60, 5)
    mdl = SimpleCNN(in_channels=60, hidden_shape=20, output_shape=1, kernel_size=8,
                    filter_size=8)

    #test1 = mdl.forward(torch.tensor(sampleX))
    test2 = mdl.forward(torch.rand(10, 60, 100))

    print(test2.shape)
    feats = load_features()

    # grab the model features and targets
    features = [f for f in feats.columns if f.startswith('feature')]
    lag_feats = [f for f in feats.columns if f.startswith('lag')]
    target = ['target']
    full = features + target + lag_feats

    # add in more
    features += lag_feats

    # condense and group
    X = feats[full].dropna()
    X.dropna(inplace=True)

    prep = PrePTestSeqData(X)
    newX, _ = split_Xy_for_seq(X[features], X['target'], step_size=20)
   
    # returns the loader
    def load_data_torch(X, y, batch_size=64):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

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


    def train_model(epoch, model, train_loader, optimizer, loss_fnc, max_norm=10**-3, clip_norm=False):
        iter_time = AverageMeter()
        losses = AverageMeter()
      
        for idx, (data, target) in enumerate(train_loader):
            start = time.time()

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

                # target needs to be reshaped as well
                #print('shape of data', data.shape)
                #print('shape of target', target.shape)


            # forward step
            out = model(data)
            out = out.cuda()
            loss = loss_fnc(out, target)

            # gradient descent step
            optimizer.zero_grad()
            loss.backward()

            # gradient norm
            if clip_norm:
                clip_grad_norm_(model.parameters(), max_norm=max_norm)

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
    model_path = 'model.pt'
    EPOCHS = 100
    learning_rate = 1e-3
    batch_size = 512
    hidden_layer_size = 5
    dropout_rate = .30
    max_norm = 0.01
    early_stopping = 25
    #reg = 1e-5

    filters = 3
    kernel_size= 10
    pool_size=2
    sequence_length = 20

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

         # initalize a scaler
         scaler = RobustScaler()

         # we only scale the 90% train X , so we don't learn the mean and sigma of the validation set
         scaler.fit(X_train2)
         X_train2 = retain_pandas_after_scale(X_train2, scaler)

         # now scaler X_train2 and Xval
         X_val = retain_pandas_after_scale(X_val, scaler)

         # Model
         model = SimpleCNN(in_channels=X_train.shape[1],
                           hidden_shape=hidden_layer_size,
                           output_shape=1, filter_size=filters,
                           kernel_size=kernel_size,
                           droput_rate=dropout_rate,
                           pool_size=pool_size)
         
         model.to(torch.device('cuda'))
         optimizer = Adam(model.parameters(), lr=learning_rate)
         loss_func = SharpeLoss(risk_trgt=.15)

         # before sending into dataloader we need to create the sequences for both train and val
         X_train2, y_train2 = split_Xy_for_seq(X_train=X_train2,
                                               y_train=y_train2,
                                               step_size=sequence_length,
                                               return_pandas=False,
                                               split_func=split_rolling_sequences_for_cnn)
         
         X_val, y_val = split_Xy_for_seq(X_train=X_val,
                                         y_train=y_val,
                                         step_size=sequence_length,
                                         return_pandas=False,
                                         split_func=split_rolling_sequences_for_cnn)
         

          # our data-loaders
         dataloader = load_data_torch(X_train2, y_train2, batch_size=batch_size)
         valdataloader = load_data_torch(X_val, y_val, batch_size=batch_size)

         early_stop_count = 0
         best_val_loss = float('inf')
         for epoch in range(EPOCHS):
             train_loss = train_model(epoch, model, dataloader, optimizer,
                                      loss_func, max_norm=max_norm, clip_norm=True)
             
             val_loss = validate_model(epoch, model, valdataloader, loss_func)

             learning_curves.loc[epoch, 'train_loss'] = train_loss
             learning_curves.loc[epoch, 'val_loss'] = val_loss

             if val_loss < best_val_loss:
                 best_val_loss = val_loss
                 torch.save(model.state_dict(), model_path)
                 early_stop_count += 0
             else:
                 early_stop_count +=1

             if early_stop_count == early_stopping:
                 print(f'Early Stopping Applied on Epoch: {epoch}')
                 break
             

         learning_curves.plot()
         plt.title(f'CV Split: {idx}')
         plt.savefig(f'learning_curves_{idx}.png')
         plt.clf()

         # scale 
         X_test2 = X_test.copy()
         X_test2 = retain_pandas_after_scale(X_test2, scaler)

         test_start, test_end = X_test.index.get_level_values('date')[0], X_test.index.get_level_values('date')[-1]
         print(f'Test Start :{test_start} | Test End :{test_end}')

         # get correct test data
         xs1 = prep.run_all_splits((test_start, test_end), newX)

         # this is where we need to start our new logic to generate test data in a shape
         # able to fit into the model
        #  xs, _ =split_Xy_for_seq(X_test2, y_test,
        #                          step_size=sequence_length,
        #                          split_func=split_rolling_sequences_for_cnn,
        #                          return_pandas=True)
         
        
         with torch.no_grad():
            model.eval()
            preds = aggregate_seq_preds(model, xs1, features=features)
            # preds = model(X_test2)
            # preds = preds.cpu().detach().numpy()
            # preds=preds.reshape(preds.shape[0], )
            # preds = pd.Series(data=preds, index=y_test.index)
            predictions.append(preds)
         break
         

    
    # 1d conv nets
    preds=pd.concat(predictions).sort_index()
    preds = preds.to_frame('Conv1D')
    feats = feats.join(preds[['Conv1D']], how='left')
    feats.dropna(subset=['Conv1D'], inplace=True)
    dates = feats.index.get_level_values('date').unique().to_list()
    strat_rets = process_jobs(dates, feats, signal_col='Conv1D')
    print(get_returns_breakout(strat_rets.fillna(0.0).to_frame('Conv1D')))



    