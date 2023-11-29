import pandas as pd
import numpy as np
import time
import copy
import matplotlib.pyplot as plt

from utils.utils import get_cv_splits, load_features, train_val_split, process_jobs, get_returns_breakout
from losses import jm_loss as L
# from losses import ryan_losses as L
# from losses import masa_loss as L

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import models as M
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import optuna

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##########################################################
# Parameters
##########################################################

criterion_name = 'SharpeLoss' # MSELoss, SharpeLossTargetOnly, WeightedSharpeLoss
model_name = 'SimpleMLP' # SimpleMLP

print(model_name, criterion_name)
##########################################################
# Training
##########################################################
filename = model_name + "_" + criterion_name

start_time = time.time()

criterion = getattr(L, criterion_name)
criterion = criterion()
print(criterion)

feats = load_features()

# grab the model features and targets
features = [f for f in feats.columns if f.startswith('feature')]
lag_feats = [f for f in feats.columns if f.startswith('lag')]
target = ['target']
full = features + target + lag_feats

# add in more
features += lag_feats

print(features)

# condense and group
X = feats[full].dropna()
X.dropna(inplace=True)

def plot_curves(train_history, valid_history, filename):
    epochs = range(len(train_history))
    plt.plot(epochs, train_history, label='train')
    plt.plot(epochs, valid_history, label='valid')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve - '+filename)
    plt.savefig('images/' + filename + '.png', bbox_inches="tight")
    plt.show()
    plt.clf()

def plot_curves_all(train_history_all, valid_history_all, filename):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        epochs = range(len(train_history_all[i]))
        ax.plot(epochs, train_history_all[i], label='train')
        ax.plot(epochs, valid_history_all[i], label='valid')
        ax.legend(loc='upper right')
        ax.set_title('Loss Curve: CV-' + str(i))
    plt.savefig('images/' + filename + '.png', bbox_inches="tight")
    plt.show()
    plt.clf()

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

def validate_model(epoch, model, val_loader, loss_fnc, gpu):
    iter_time = AverageMeter()
    losses = AverageMeter()

    for idx, (data, target) in enumerate(val_loader):
        start = time.time()

        if torch.cuda.is_available:
            data = data.to(torch.device(gpu))
            target = target.to(torch.device(gpu))

        with torch.no_grad():
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

def train_model(epoch, model, train_loader, optimizer, loss_fnc, clip, gpu):
    iter_time = AverageMeter()
    losses = AverageMeter()

    for idx, (data, target) in enumerate(train_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.to(torch.device(gpu))
            target = target.to(torch.device(gpu))

        # forward step
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

# def run_train(params):
#
#     batch_size = params['batch_size']
#     epochs = 100
#     patience = 25
#     learning_rate = params['learning_rate']
#     maximum_gradient_norm = params['maximum_gradient_norm']
#
#     model_params = {
#         'input_dim': 10*6,
#         'hidden_size': params['hidden_size'],
#         'dropout': params['dropout'],
#         'num_classes': 1,
#     }
#
#     # our data-loaders
#     dataloader = load_data_torch(X_train2, y_train2, batch_size=batch_size)
#     valdataloader = load_data_torch(X_val, y_val, batch_size=batch_size)
#
#     model = getattr(M, model_name)
#     model = model(**model_params)
#     # model = nn.DataParallel(model)
#     model.to(device)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#     best_val_loss = float("inf")
#     best_model = None
#     train_history = []
#     valid_history = []
#     for epoch in range(epochs):
#
#         train_loss = train_model(epoch, model, dataloader, optimizer, criterion, maximum_gradient_norm)
#         val_loss = validate_model(epoch, model, valdataloader, criterion)
#
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model = copy.deepcopy(model)
#             patience_count = 0
#         else:
#             patience_count += 1
#
#         train_history.append(train_loss)
#         valid_history.append(val_loss)
#         # print(epoch, "Training Loss: %.4f. Validation Loss: %.4f. " % (train_loss, val_loss))
#
#         if patience_count == patience:
#             break
#
#     return best_val_loss, best_model, train_history, valid_history
# def objective(trial):
#
#     batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])
#     learning_rate = trial.suggest_categorical('learning_rate', [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0])
#     maximum_gradient_norm = trial.suggest_categorical('maximum_gradient_norm', [10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1])
#     hidden_size = trial.suggest_categorical('hidden_size', [5, 10, 20, 40, 80])
#     dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])
#
#     params = {
#         'batch_size': batch_size,
#         'learning_rate': learning_rate,
#         'maximum_gradient_norm': maximum_gradient_norm,
#         'hidden_size': hidden_size,
#         'dropout': dropout,
#     }
#
#     best_val_loss, best_model, train_history, valid_history = run_train(params)
#     return best_val_loss

def run_train(params, gpu):

    batch_size = params['batch_size']
    epochs = 100
    patience = 25
    learning_rate = params['learning_rate']
    maximum_gradient_norm = params['maximum_gradient_norm']

    model_params = {
        'input_dim': 10*6,
        'hidden_size': params['hidden_size'],
        'dropout': params['dropout'],
        'num_classes': 1,
    }

    # our data-loaders
    dataloader = load_data_torch(X_train2, y_train2, batch_size=batch_size)
    valdataloader = load_data_torch(X_val, y_val, batch_size=batch_size)

    model = getattr(M, model_name)
    model = model(**model_params)
    # model = nn.DataParallel(model)
    model.to(gpu)
    # model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_model = None
    train_history = []
    valid_history = []
    for epoch in range(epochs):

        train_loss = train_model(epoch, model, dataloader, optimizer, criterion, maximum_gradient_norm, gpu)
        val_loss = validate_model(epoch, model, valdataloader, criterion, gpu)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            patience_count = 0
        else:
            patience_count += 1

        train_history.append(train_loss)
        valid_history.append(val_loss)
        # print(epoch, "Training Loss: %.4f. Validation Loss: %.4f. " % (train_loss, val_loss))

        if patience_count == patience:
            break

    return best_val_loss, best_model, train_history, valid_history

##### multi-gpu processing
# Source: https://stackoverflow.com/questions/61763206/is-there-a-way-to-pass-arguments-to-multiple-jobs-in-optuna
from contextlib import contextmanager
import multiprocessing
N_GPUS = 1
class GpuQueue:

    def __init__(self):
        self.queue = multiprocessing.Manager().Queue()
        all_idxs = list(range(N_GPUS)) if N_GPUS > 0 else [None]
        for idx in all_idxs:
            self.queue.put(idx)

    @contextmanager
    def one_gpu_per_process(self):
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)

class Objective:

    def __init__(self, gpu_queue: GpuQueue):
        self.gpu_queue = gpu_queue

    def __call__(self, trial):
        with self.gpu_queue.one_gpu_per_process() as gpu_i:
            batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])
            learning_rate = trial.suggest_categorical('learning_rate', [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0])
            maximum_gradient_norm = trial.suggest_categorical('maximum_gradient_norm',
                                                              [10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1])
            hidden_size = trial.suggest_categorical('hidden_size', [5, 10, 20, 40, 80])
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])

            params = {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'maximum_gradient_norm': maximum_gradient_norm,
                'hidden_size': hidden_size,
                'dropout': dropout,
            }

            best_val_loss, best_model, train_history, valid_history = run_train(params, gpu=gpu_i)
            return best_val_loss
def run_hyper_parameter_tuning():
    # sampler = optuna.samplers.TPESampler(seed=42)
    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(sampler=sampler)
    study.optimize(Objective(GpuQueue()), n_trials=10, n_jobs=10, show_progress_bar=True)
    # study.optimize(objective, n_trials=50, n_jobs=-1, show_progress_bar=True)
    # study.optimize(objective, n_trials=2, show_progress_bar=True)

    best_params = study.best_params

    # export the best parameters
    txt_results = open("best_params_mlp.txt", "a")
    print(f"Best Parameters: {best_params}", file=txt_results)
    txt_results.close()

    print("Best Parameters:", best_params)

    return best_params

# set up a base model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

predictions = []
scores = []
print("get_cv_splits:", get_cv_splits(X))
cv_count = 0
train_history_all = []
valid_history_all = []
for train, test in tqdm(get_cv_splits(X)):
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
    best_val_loss, best_model, train_history, valid_history = run_train(best_params, 'cuda')

    print('Best Loss: {:.4f}'.format(best_val_loss))
    train_history_all.append(train_history)
    valid_history_all.append(valid_history)
    # plot_curves(train_history, valid_history, filename + "_cv-" + str(cv_count))
    cv_count += 1

    # scale
    X_test2 = X_test.copy()
    X_test2 = scaler.transform(X_test2)

    X_test2 = torch.tensor(X_test2, dtype=torch.float32)
    X_test2 = X_test2.cuda()

    with torch.no_grad():
        best_model.eval()
        preds = best_model(X_test2)
        preds = preds.cpu().detach().numpy()
        preds = preds.reshape(preds.shape[0], )
        preds = pd.Series(data=preds, index=y_test.index)
        predictions.append(preds)

plot_curves_all(train_history_all, valid_history_all, filename + "_cv-all")

preds=pd.concat(predictions).sort_index()
preds = preds.to_frame('mlp')
feats = feats.join(preds[['mlp']], how='left')
feats.dropna(subset=['mlp'], inplace=True)
dates = feats.index.get_level_values('date').unique().to_list()

print("*** running back-test ***")
strat_rets = process_jobs(dates, feats, signal_col='mlp')
ret_breakout = get_returns_breakout(strat_rets.fillna(0.0).to_frame('mlp_bench'))
ret_breakout.to_csv('results/' + filename + '.csv')
print(ret_breakout)

ax = strat_rets.fillna(0.).cumsum().plot()
ax.figure.savefig('images/' + filename + '_timeline.png')

print("time:", (time.time() - start_time) / 60)