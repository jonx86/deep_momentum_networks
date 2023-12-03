import matplotlib.pyplot as plt
import pandas as pd
import copy

from sklearn.preprocessing import RobustScaler

import time

import models as M
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from losses import jm_loss as L
from utils.utils import load_features, get_cv_splits, train_val_split, process_jobs, get_returns_breakout
import optuna
import numpy as np
import os, glob, sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

torch.manual_seed(0)

##########################################################
# Parameters
##########################################################

# Example: python main_mlp_optuna.py MLP SharpeLoss cuda:0 50

model_name = sys.argv[1]
loss_func_name = sys.argv[2]
gpu = sys.argv[3]
n_trials = int(sys.argv[4])

batch_size_space = [256, 512, 1024, 2048]
learning_rate_space = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0]
maximum_gradient_norm_space = [10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1]
reg_space = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1]

print(model_name, loss_func_name, gpu, n_trials)
filename = model_name + "_" + loss_func_name
outfile = open("results/" + filename + ".txt", "w")
outfile_best_param = open("results/best_params_" + filename + ".txt", "w")
model_name = model_name + loss_func_name

# Parameters
model_path = 'model_weights/' + filename + '_model_'
cv_global = 0
##########################################################
# Training
##########################################################

start_time = time.time()
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
def load_data_torch(X, y, batch_size=64):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32)

    # send to cuda
    X.to(torch.device(gpu))
    y.to(torch.device(gpu))

    loader = DataLoader(list(zip(X, y)), shuffle=False, batch_size=batch_size)
    return loader


def validate_model(epoch, model, val_loader, loss_fnc):
    iter_time = AverageMeter()
    losses = AverageMeter()

    for idx, (data, target) in enumerate(val_loader):
        start = time.time()

        if torch.cuda.is_available:
            data = data.to(torch.device(gpu))
            target = target.to(torch.device(gpu))

        with torch.no_grad():
            model.eval()
            out = model(data)
            out = out.to(torch.device(gpu))
            loss = loss_fnc(out, target)

        losses.update(loss.item(), out.shape[0])
        iter_time.update(time.time() - start)

        # if idx % 10==0:
        #         print(('Epoch: [{0}][{1}/{2}]\t'
        #         'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
        #         'Loss {loss.val:.4f} ({loss.avg:.4f})\t').format(
        #             epoch,
        #             idx,
        #             len(val_loader),
        #             iter_time=iter_time,
        #             loss=losses))

    return losses.avg


def train_model(epoch, model, train_loader, optimizer, loss_fnc, clip):
    iter_time = AverageMeter()
    losses = AverageMeter()

    for idx, (data, target) in enumerate(train_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.to(torch.device(gpu))
            target = target.to(torch.device(gpu))

        # forward step
        model.train()
        out = model(data)
        out = out.to(torch.device(gpu))
        loss = loss_fnc(out, target)

        # gradient descent step
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        losses.update(loss.item(), out.shape[0])
        iter_time.update(time.time() - start)

        # if idx % 10 == 0:
        #         print(('Epoch: [{0}][{1}/{2}]\t'
        #         'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
        #         'Loss {loss.val:.4f} ({loss.avg:.4f})\t').format(
        #             epoch,
        #             idx,
        #             len(train_loader),
        #             iter_time=iter_time,
        #             loss=losses))
    return losses.avg

def run_train(params):

    batch_size = params['batch_size']
    epochs = 100
    early_stopping = 25
    learning_rate = params['learning_rate']
    maximum_gradient_norm = params['maximum_gradient_norm']
    reg = params['reg']

    model_params = {
        'input_dim': 10*6,
        'device': gpu,
    }

    # our data-loaders
    dataloader = load_data_torch(X_train2, y_train2, batch_size=batch_size)
    valdataloader = load_data_torch(X_val, y_val, batch_size=batch_size)

    # model
    model = getattr(M, model_name)
    model = model(**model_params)
    model.to(gpu)
    # print(model)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

    early_stop_count = 0
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train_model(epoch, model, dataloader, optimizer, loss_func, maximum_gradient_norm)
        val_loss = validate_model(epoch, model, valdataloader, loss_func)

        learning_curves.loc[epoch, 'train_loss'] = train_loss
        learning_curves.loc[epoch, 'val_loss'] = val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # best_model = copy.deepcopy(model)
            early_stop_count = 0
        else:
            early_stop_count += 1

        # print(epoch, "Training Loss: %.4f. Validation Loss: %.4f. " % (train_loss, val_loss))

        if early_stop_count == early_stopping:
            break

    # return best_val_loss, best_model
    return best_val_loss, model

def objective(trial):

    batch_size = trial.suggest_categorical('batch_size', batch_size_space)
    learning_rate = trial.suggest_categorical('learning_rate', learning_rate_space)
    maximum_gradient_norm = trial.suggest_categorical('maximum_gradient_norm', maximum_gradient_norm_space)
    reg = trial.suggest_categorical('reg', reg_space)

    params = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'maximum_gradient_norm': maximum_gradient_norm,
        'reg': reg,
    }

    best_val_loss, best_model = run_train(params)
    return best_val_loss
def run_hyper_parameter_tuning():
    sampler = optuna.samplers.TPESampler(seed=42)
    # sampler = optuna.samplers.RandomSampler(seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_trials, show_progress_bar=True)

    best_params = study.best_params

    print("Best Parameters:", best_params)

    return best_params

loss_func = getattr(L, loss_func_name)
loss_func = loss_func()
print(loss_func)

# Dataset
dataset = load_features()
dataset.dropna(inplace=True)

# Future IDs
futures = dataset.index.get_level_values('future').unique().tolist()

# Inputs and Targets
features = [column for column in dataset.columns if column.startswith('feature')]
lags = [column for column in dataset.columns if column.startswith('lag')]
features = features + lags
target = ['target']

# Workingset
X = dataset[features + target]

device = torch.device('cuda')

for modelname in glob.glob(model_path + "*"):
    print("removed", modelname)
    os.remove(modelname)

predictions = [0] * 6
train_history_all = []
valid_history_all = []
# now start the loop
for idx, (train, test) in enumerate(get_cv_splits(X)):
    print("cv run:", idx)
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

    best_params = run_hyper_parameter_tuning()
    best_val_loss, best_model = run_train(best_params)

    print('Best Loss: {:.4f}'.format(best_val_loss))
    print('Best Params:', best_params)
    learning_curves.plot()
    plt.title(f'CV Split: {idx}')
    plt.savefig('images/' + filename + f'_learning_curves_{idx}.png')
    plt.clf()

    # export the best parameters
    print('cv: ', cv_global, file=outfile_best_param)
    print(best_params, file=outfile_best_param, flush=True)

    # scale
    X_test2 = X_test.copy()
    X_test2 = scaler.transform(X_test2)

    X_test2 = torch.tensor(X_test2, dtype=torch.float32)
    X_test2 = X_test2.to(torch.device(gpu))

    with torch.no_grad():
        best_model.eval()
        preds = best_model(X_test2)
        preds = preds.cpu().detach().numpy()
        preds=preds.reshape(preds.shape[0], )
        if loss_func_name == 'RegressionLoss':
            preds = np.sign(preds)
        if loss_func_name == 'BinaryClassificationLoss':
            preds -= .50
            preds = np.sign(preds)
        preds = pd.Series(data=preds, index=y_test.index)
        predictions[idx] = preds

    cv_global += 1
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
    print(values)

preds=pd.concat(predictions).sort_index()
preds = preds.to_frame('mlp')
feats = dataset.join(preds[['mlp']], how='left')
feats.dropna(subset=['mlp'], inplace=True)
dates = feats.index.get_level_values('date').unique().to_list()
strat_rets = process_jobs(dates, feats, signal_col='mlp')
ret_breakout = get_returns_breakout(strat_rets.fillna(0.0).to_frame('mlp_bench'))
print(ret_breakout)
print('Final', file=outfile, flush=True)
print(ret_breakout, file=outfile, flush=True)

outfile.close()
outfile_best_param.close()

ax = strat_rets.fillna(0.).cumsum().plot()
ax.figure.savefig('images/' + filename + '_timeline.png')

print("time:", (time.time() - start_time) / 60)