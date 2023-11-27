import pandas as pd
import numpy as np
import time
import copy
import matplotlib.pyplot as plt

from utils.utils import get_cv_splits, load_features, train_val_split, process_jobs, get_returns_breakout
from losses import masa_loss as L

from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import models as M
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##########################################################
# Parameters
##########################################################

batch_size = 128 * 8 * 8
epochs = 100
learning_rate = 1e-5
momentum = 0.
reg = 0.0005
criterion_name = 'SharpeLossTargetOnly' # MSELoss, SharpeLossTargetOnly
model_name = 'SimpleMLP'
patience = 10

print(model_name, criterion_name, learning_rate)
print("batch_size:", batch_size)
##########################################################
# Training
##########################################################

filename = model_name + "_" + criterion_name

start_time = time.time()

criterion = getattr(L, criterion_name)
criterion = criterion()
print(criterion)

# load the data
feats = load_features()

# grab the model features and targets
features = [f for f in feats.columns if f.startswith('feature')]
target = ['target']

# get all
full = features + target

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
    plt.savefig('images/' + filename + '.png')
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
    plt.savefig('images/' + filename + '.png')
    plt.show()
    plt.clf()

def run_train(epoch, X_train_loader, y_train_loader, model, optimizer, criterion):
    total_loss = 0.

    for idx, (data, target) in enumerate(zip(X_train_loader, y_train_loader)):

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss, total_loss / len(X_train_loader)

def run_validate(epoch, X_val_loader, y_val_loader, model, criterion):

    total_loss = 0.
    for idx, (data, target) in enumerate(zip(X_val_loader, y_val_loader)):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        out = model(data)
        loss = criterion(out, target)

        total_loss += loss.item()

    return total_loss, total_loss / len(X_val_loader)

# set up a base model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

predictions = []
scores = []
print("get_cv_splits:", get_cv_splits(X))
cv_count = 0
train_history_all = []
valid_history_all = []
for train, test in tqdm(get_cv_splits(X)):
    # break out X and y train, test
    X_train_tmp, y_train_tmp = train[features], train[target]
    X_test_tmp, y_test_tmp = test[features], test[target]
    X_train = torch.tensor(X_train_tmp.values).float()
    y_train = torch.tensor(y_train_tmp.values).float()
    X_test = torch.tensor(X_test_tmp.values).float()
    y_test = torch.tensor(y_test_tmp.values).float()

    X_train_loader = DataLoader(X_train,
                              batch_size=batch_size,
                              shuffle=False)
    y_train_loader = DataLoader(y_train,
                              batch_size=batch_size,
                              shuffle=False)

    X_test_loader = DataLoader(X_test,
                              batch_size=batch_size,
                              shuffle=False)
    y_test_loader = DataLoader(y_test,
                              batch_size=batch_size,
                              shuffle=False)

    model = getattr(M, model_name)
    model = model()
    model = nn.DataParallel(model)
    model.to(device)

    # optimizer = torch.optim.SGD(model.parameters(),
    #                             learning_rate,
    #                             momentum=momentum,
    #                             weight_decay=reg)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best = float("inf")
    best_cm = None
    best_model = None
    train_history = []
    valid_history = []
    for epoch in range(epochs):

        # train loop
        train_loss, avg_train_loss  = run_train(epoch, X_train_loader, y_train_loader, model, optimizer, criterion)

        # validation loop
        val_loss, avg_val_loss = run_validate(epoch, X_test_loader, y_test_loader, model, criterion)

        if avg_val_loss < best:
            best = avg_val_loss
            best_model = copy.deepcopy(model)
            patience_count = 0
        else:
            patience_count += 1

        train_history.append(avg_train_loss)
        valid_history.append(avg_val_loss)
        print(epoch, "Training Loss: %.4f. Validation Loss: %.4f. " % (avg_train_loss, avg_val_loss))

        if patience_count == patience:
            break

    print('Best Loss: {:.4f}'.format(best))
    train_history_all.append(train_history)
    valid_history_all.append(valid_history)
    plot_curves(train_history, valid_history, filename + "_cv-" + str(cv_count))
    cv_count += 1

    # now fit the best model
    if torch.cuda.is_available():
        X_test = X_test.cuda()
    preds = best_model(X_test)

    # append the predictions
    predictions.append(pd.Series(index=y_test_tmp.index, data=preds.cpu().detach().numpy().ravel()))

    # score
    scores.append(mean_squared_error(y_test_tmp, preds.cpu().detach().numpy().ravel()))

plot_curves_all(train_history_all, valid_history_all, filename + "_cv-all")

# print the scores, concat the predictions , to feed into our back-test code
predictions = pd.concat(predictions).sort_index()
scores = np.array(scores)

# create the predictions
predictions = predictions.to_frame('rf_bm')
predictions['rf_bin_bm'] = np.sign(predictions['rf_bm'])

# join back in binary-predictions to feats
feats = feats.join(predictions[['rf_bin_bm']], how='left')
feats.dropna(subset=['rf_bin_bm'], inplace=True)

# signal column
signal_col = 'rf_bin_bm'

# dates
dates = feats.index.get_level_values('date').unique().to_list()

# need a 252 day warm up
dates = dates[252:]

# back-test, signal_col is the prediction for X, in this case just binary -1, or 1
print("*** running back-test ***")
strat_rets = process_jobs(dates, feats, signal_col='rf_bin_bm')
ret_breakout = get_returns_breakout(strat_rets.fillna(0.0).to_frame('rf_benchmark'))
ret_breakout.to_csv('results/' + filename + '.csv')
print(ret_breakout)

ax = strat_rets.fillna(0.).cumsum().plot()
ax.figure.savefig('images/' + filename + '_timeline.png')

print("time:", (time.time() - start_time) / 60)