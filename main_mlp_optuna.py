import matplotlib.pyplot as plt
import pandas as pd
import copy
import numpy as np
from datetime import datetime
from pytz import timezone
import pytz

from sklearn.preprocessing import RobustScaler

import time

import models as M
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from losses import jm_loss as L
from utils.utils import *
import optuna
import numpy as np
import os, glob, sys
import random

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

############## SET SEED ##############
def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(0)

##########################################################
# Parameters
##########################################################

# Example: python main_mlp_optuna.py MLP SharpeLoss cuda:0 50

model_name = sys.argv[1]
loss_func_name = sys.argv[2]
gpu = sys.argv[3]
n_trials = int(sys.argv[4])
n_jobs = int(sys.argv[5])
max_epochs = int(sys.argv[6])
sampler_type = sys.argv[7]

# batch_size_space = [128, 256, 512, 1024]
batch_size_space = [256, 512, 1024, 2048]
hidden_size_space = [10, 20, 40]
weight_space = [0.6, 0.55, 0.5, 0.45, 0.4]

# batch_size_space = [2048]
# learning_rate_space = [0.001]
# maximum_gradient_norm_space = [0.01]
# hidden_size_space = [20]
# dropout_space = [0.3]

print(model_name, loss_func_name, gpu, n_trials, n_jobs, max_epochs)

##########################################################
# Create directory to store data
##########################################################
filename = model_name + "_" + loss_func_name + "_" + sampler_type + "_" + str(max_epochs)

now = datetime.now(tz=pytz.utc)
now = now.astimezone(timezone('US/Pacific'))
data_str = now.strftime("%m%d") + "_" + now.strftime('%H%M%S')
loc_files = "files/" + filename + "_" + data_str + "/"
if not os.path.exists(loc_files):
    os.makedirs(loc_files)

outfile = open(loc_files + filename + ".txt", "w")
outfile_best_param = open(loc_files + "best_params_" + filename + ".txt", "w")
model_name = model_name + loss_func_name

model_path = loc_files + filename + '_model_'
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
    global global_val_loss
    global learning_curves_global

    batch_size = params['batch_size']
    epochs = max_epochs
    early_stopping = 25
    learning_rate = params['learning_rate']
    maximum_gradient_norm = params['maximum_gradient_norm']

    model_params = {
        'input_dim': 8*6,
        'hidden_size': params['hidden_size'],
        'dropout': params['dropout'],
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

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, verbose=False)
    loss_func = getattr(L, loss_func_name)
    if loss_func_name == "SharpeLossCustom":
        loss_func = loss_func(params['weight'])
    else:
        loss_func = loss_func()

    early_stop_count = 0
    best_val_loss = float('inf')
    learning_curves = pd.DataFrame(columns=['train_loss', 'val_loss'])
    for epoch in range(epochs):
        train_loss = train_model(epoch, model, dataloader, optimizer, loss_func, maximum_gradient_norm)
        val_loss = validate_model(epoch, model, valdataloader, loss_func)
        scheduler.step(val_loss)

        learning_curves.loc[epoch, 'train_loss'] = train_loss
        learning_curves.loc[epoch, 'val_loss'] = val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
        else:
            early_stop_count += 1

        # if val_loss < global_val_loss:
        #     global_val_loss = val_loss

        # print(epoch, "Training Loss: %.4f. Validation Loss: %.4f. " % (train_loss, val_loss))

        if early_stop_count == early_stopping:
            break

    if val_loss < global_val_loss:
        global_val_loss = val_loss

    # if global_val_loss == best_val_loss:
    if global_val_loss == val_loss:
        best_model = copy.deepcopy(model)
        torch.save(best_model, model_path + str(idx) + '.pt')
        best_model_state = copy.deepcopy(best_model.state_dict())
        torch.save(best_model_state, model_path + str(idx) + '_state_dict.pt')
        learning_curves_global = learning_curves

    return best_val_loss

def objective(trial):

    batch_size = trial.suggest_categorical('batch_size', batch_size_space)
    hidden_size = trial.suggest_categorical('hidden_size', hidden_size_space)
    # learning_rate = trial.suggest_loguniform("learning_rate", 10 ** -5, 10 ** -1)
    learning_rate = trial.suggest_loguniform("learning_rate", 10 ** -4, 10 ** -2)
    maximum_gradient_norm = trial.suggest_loguniform("maximum_gradient_norm", 10 ** -3, 10 ** -1)
    dropout = trial.suggest_uniform("dropout", 0.2, 0.4)

    params = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'maximum_gradient_norm': maximum_gradient_norm,
        'hidden_size': hidden_size,
        'dropout': dropout,
    }

    if loss_func_name == "SharpeLossCustom":
        weight = trial.suggest_categorical('weight', weight_space)
        params['weight'] = weight

    best_val_loss = run_train(params)
    return best_val_loss
def run_hyper_parameter_tuning():
    if sampler_type == 'TPE':
        sampler = optuna.samplers.TPESampler(seed=0)
    elif sampler_type == 'Random':
        sampler = optuna.samplers.RandomSampler(seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    best_params = study.best_params

    print("Best Parameters:", best_params)

    return best_params

data = load_features()
target = ['target']
both = MLP_FEATURES+target

full = data[both].dropna(subset=both)
X = full[both]

for modelname in glob.glob(model_path + "*"):
    print("removed", modelname)
    os.remove(modelname)

predictions = [0] * 6
train_history_all = []
valid_history_all = []
# now start the loop
for idx, (train, test) in enumerate(get_cv_splits(X)):
    print("cv run:", idx)
    learning_curves_global = pd.DataFrame(columns=['train_loss', 'val_loss'])
    iter_time = AverageMeter()
    train_losses = AverageMeter()

    # break out X and y train
    X_train, y_train = train[MLP_FEATURES], train[target]
    X_test, y_test = test[MLP_FEATURES], test[target]

    # validation split
    X_train2, X_val, y_train2, y_val = train_val_split(X_train, y_train)

    scaler = RobustScaler()

    # we only scale the 90% train X , so we don't learn the mean and sigma of the validation set
    X_train2 = scaler.fit_transform(X_train2)

    # now scaler X_train2 and Xval
    X_val = scaler.transform(X_val)

    global_val_loss = float('inf')
    best_params = run_hyper_parameter_tuning()
    print('Best Params:', best_params)
    print('Global Val Loss:', global_val_loss)
    model = torch.load(model_path + str(idx) + '.pt')
    model.load_state_dict(torch.load(model_path + str(idx) + '_state_dict.pt'))
    model.to(gpu)

    learning_curves_global.plot()
    plt.title(f'CV Split: {idx}')
    plt.savefig(loc_files + filename + f'_learning_curves_{idx}.png')
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
        model.eval()
        preds = model(X_test2)
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
    preds = pd.concat([predictions[idx]]).sort_index()
    preds = preds.to_frame(model_name)
    new_dataset = data.copy()
    feats = new_dataset.join(preds[[model_name]], how='left')
    feats.dropna(subset=[model_name], inplace=True)
    dates = feats.index.get_level_values('date').unique().to_list()
    strat_rets = process_jobs(dates, feats, signal_col=model_name)
    values = get_returns_breakout(strat_rets.fillna(0.0).to_frame(model_name + '_bench'))
    print('idx:', idx, 'best_val_loss:', global_val_loss, file=outfile)
    print(values, file=outfile, flush=True)
    print(values)

preds = pd.concat(predictions).sort_index()
preds = preds.to_frame(model_name)
feats = data.join(preds[[model_name]], how='left')
feats.dropna(subset=[model_name], inplace=True)
dates = feats.index.get_level_values('date').unique().to_list()
strat_rets = process_jobs(dates, feats, signal_col=model_name)
ret_breakout = get_returns_breakout(strat_rets.fillna(0.0).to_frame(model_name + '_bench'))
print(ret_breakout)
print('Final', file=outfile, flush=True)
print(ret_breakout, file=outfile, flush=True)

outfile.close()
outfile_best_param.close()

strat_rets.to_pickle(loc_files + "strat_rets_" + filename + ".pkl")

ax = strat_rets.fillna(0.).cumsum().plot()
ax.figure.savefig(loc_files + filename + '_timeline.png')

print(data_str)
print("total time:", (time.time() - start_time) / 60)