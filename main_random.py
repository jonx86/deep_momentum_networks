import pandas as pd
import numpy as np
import time

from utils.utils import get_cv_splits, load_features, train_val_split, process_jobs, get_returns_breakout
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import torch

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

filename = "random_baseline"

start_time = time.time()

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

    preds = y_test[torch.randperm(X_test.size()[0])]

    # append the predictions
    predictions.append(pd.Series(index=y_test_tmp.index, data=preds.cpu().detach().numpy().ravel()))

    # score
    scores.append(mean_squared_error(y_test_tmp, preds.cpu().detach().numpy().ravel()))

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
ax.figure.savefig("images/" + filename + '_timeline.png')

print("time:", (time.time() - start_time) / 60)