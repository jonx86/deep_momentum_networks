import matplotlib.pyplot as plt
import pandas as pd
import copy

from sklearn.preprocessing import RobustScaler

import time

import models as M
import torch
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler

from losses import jm_loss as L
from utils.utils import *

import optuna
import os, glob, sys

import warnings
warnings.filterwarnings("ignore")

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

batch_size_space = [256, 512, 1024, 2048]
learning_rate_space = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0]
maximum_gradient_norm_space = [10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1]
hidden_size_space = [5, 10, 20, 40, 80]
dropout_space = [0.1, 0.2, 0.3, 0.4, 0.5]

# batch_size_space = [2048]
# learning_rate_space = [0.001]
# maximum_gradient_norm_space = [0.01]
# hidden_size_space = [20]
# dropout_space = [0.3]

print(model_name, loss_func_name, gpu, n_trials, n_jobs)
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

def run_train(params):

    batch_size = params['batch_size']
    epochs = 100
    early_stopping = 25
    in_channels = 60
    out_channels = 3
    kernal_size = 2
    dilations = [5, 10, 15, 21, 42]
    learning_rate = params['learning_rate']
    maximum_gradient_norm = params['maximum_gradient_norm']

    model_params = {
        'in_channels': in_channels,
        'out_channels': out_channels,
        'kernal_size': kernal_size,
        'num_layers': dilations,
        'num_stacks': 1,
        'hidden_size': params['hidden_size'],
        'dropOutRate': params['dropout'],
        'device': gpu
    }

    train_loader = load_data_torch(X_train2, y_train2,
                                   batch_size=batch_size,
                                   device=gpu)

    val_loader = load_data_torch(X_val, y_val,
                                 batch_size=batch_size,
                                 device=gpu)

    # model
    model = getattr(M, model_name)
    model = model(**model_params)
    model.to(gpu)
    # print(model)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, verbose=True)
    loss_func = getattr(L, loss_func_name)
    loss_func = loss_func()

    early_stop_count = 0
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train_model(epoch,
                                 model,
                                 train_loader=train_loader,
                                 optimizer=optimizer,
                                 loss_fnc=loss_func,
                                 clip_norm=True,
                                 max_norm=maximum_gradient_norm,
                                 device=gpu)

        val_loss = validate_model(epoch,
                                  model,
                                  val_loader,
                                  loss_fnc=loss_func,
                                  device=gpu)
        scheduler.step(val_loss)

        learning_curves.loc[epoch, 'train_loss'] = train_loss
        learning_curves.loc[epoch, 'val_loss'] = val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
        else:
            early_stop_count += 1

        global global_val_loss
        if val_loss < global_val_loss:
            global_val_loss = val_loss
            best_model = copy.deepcopy(model)
            torch.save(best_model, model_path + str(idx) + '.pt')
            best_model_state = copy.deepcopy(best_model.state_dict())
            torch.save(best_model_state, model_path + str(idx) + '_state_dict.pt')

        # print(epoch, "Training Loss: %.4f. Validation Loss: %.4f. " % (train_loss, val_loss))

        if early_stop_count == early_stopping:
            break

    return best_val_loss

def objective(trial):

    batch_size = trial.suggest_categorical('batch_size', batch_size_space)
    learning_rate = trial.suggest_categorical('learning_rate', learning_rate_space)
    maximum_gradient_norm = trial.suggest_categorical('maximum_gradient_norm', maximum_gradient_norm_space)
    hidden_size = trial.suggest_categorical('hidden_size', hidden_size_space)
    dropout = trial.suggest_categorical('dropout', dropout_space)

    params = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'maximum_gradient_norm': maximum_gradient_norm,
        'hidden_size': hidden_size,
        'dropout': dropout,
    }

    best_val_loss = run_train(params)
    return best_val_loss

def run_hyper_parameter_tuning():
    # sampler = optuna.samplers.TPESampler(seed=42)
    sampler = optuna.samplers.RandomSampler(seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    best_params = study.best_params

    print("Best Parameters:", best_params)

    return best_params

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

for modelname in glob.glob(model_path + "*"):
    print("removed", modelname)
    os.remove(modelname)

NUM_CORES = -1  # -1 for all cores, there are 3 multi-processed data aggregate functions, because we need to operate on the future level

# used for getting the correct batching of the test set to feed into LSTM
prep = PrePTestSeqData(X)

# TODO split Xy for seq could to be multi-processed for now this is slow but works - Use joblib
# My method to process the entire dataset first so I can filter on correct dates for test set and
# don't need to look back a small window into train set

SEC_LEN = 63
MODEL_NAME = 'WaveNet'
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
    print("cv run:", idx)
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

    global_val_loss = float('inf')
    best_params = run_hyper_parameter_tuning()
    print('Best Params:', best_params)
    print('Global Val Loss:', global_val_loss)
    model = torch.load(model_path + str(idx) + '.pt')
    model.load_state_dict(torch.load(model_path + str(idx) + '_state_dict.pt'))
    model.to(gpu)

    learning_curves.plot()
    plt.title(f'CV Split: {idx}')
    plt.savefig('images/' + filename + f'_learning_curves_{idx}.png')
    plt.clf()

    # export the best parameters
    print('cv: ', cv_global, file=outfile_best_param)
    print(best_params, file=outfile_best_param, flush=True)

    # we need a tuple of the test set start and end dates
    test_start, test_end = X_test.index.get_level_values('date')[0], X_test.index.get_level_values('date')[-1]
    print(f'Test Start :{test_start} | Test End :{test_end}')

    # we don't need X train anymore
    del X_train

    xs1 = mpSplits(prep.split_single_future,
                   (test_start, test_end), newX,
                   n_jobs=NUM_CORES)

    with torch.no_grad():
        model.eval()
        # feed in sequences for each future and get the predictions, take just the last time-step
        preds = aggregate_seq_preds(model, xs1,
                                    features=features,
                                    device=gpu,
                                    lstm=False,
                                    seq_out=False,
                                    n_jobs=2)

        preds = preds.to_frame(model_name)
        predictions.append(preds)

    cv_global += 1
    preds=pd.concat([predictions[idx]]).sort_index()
    new_dataset = data.copy()
    feats = new_dataset.join(preds[[model_name]], how='left')
    feats.dropna(subset=[model_name], inplace=True)
    dates = feats.index.get_level_values('date').unique().to_list()
    strat_rets = process_jobs(dates, feats, signal_col=model_name)
    values = get_returns_breakout(strat_rets.fillna(0.0).to_frame(model_name + '_bench'))
    print('idx: ', idx, file=outfile)
    print(values, file=outfile, flush=True)
    print(values)

preds=pd.concat(predictions).sort_index()
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

strat_rets.to_pickle("results/strat_rets_" + filename + ".pkl")

ax = strat_rets.fillna(0.).cumsum().plot()
ax.figure.savefig('images/' + filename + '_timeline.png')

print("total time:", (time.time() - start_time) / 60)