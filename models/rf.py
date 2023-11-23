import pandas as pd
import numpy as np
from sklearn.ensemble import  RandomForestRegressor

from utils.utils import (get_cv_splits,
                         load_features,
                         train_val_split,
                         process_jobs,
                         get_returns_breakout)

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterSampler
from tqdm import tqdm

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

# set up a base model
baseRF = RandomForestRegressor(max_depth=5, 
                               n_estimators=1000,
                               max_features=int(1),
                               n_jobs=10)

# simple-grid
grid = {'n_estimators': np.arange(100, 2000, 100),
        'max_depth': np.arange(2, 12, 1),
        'max_features': [int(1), 'sqrt'],
        'min_weight_fraction_leaf': np.arange(0.0, 0.05, 0.005)}

params = ParameterSampler(n_iter=25, param_distributions=grid)

# NOTE - we can either frame as a regression or classification problem in this example we use regression
predictions = []
scores = []
for train, test in tqdm(get_cv_splits(X)):
    # break out X and y train, test
    X_train, y_train = train[features], train[target] 
    X_test, y_test = test[features], test[target]

    # hyper-param loop
    X_train2, X_val, y_train2, y_val = train_val_split(X_train, y_train)
    print(X_train2.shape, X_val.shape)

    # inner loop for parameter tuning
    gscv_scores = {'scores': [], 'grid':[]}
    for k, p in enumerate(params):
        model = RandomForestRegressor(**p)
        model.n_jobs=-1
        model.fit(X_train2, y_train2.values.reshape(y_train2.shape[0], ))
        _pred = model.predict(X_val)
        _score = mean_squared_error(y_val, _pred)
        gscv_scores['scores'].append(_score)
        gscv_scores['grid'].append(p)
        print(f'Iter: {k}: Score: {_score}')

    # now fit the best model
    best_model = pd.DataFrame(gscv_scores).sort_values(by='scores').head(1)['grid'].values[0]
    print(best_model)
    best_model = RandomForestRegressor(**best_model)
    best_model.fit(X_train, y_train.values.reshape(y_train.shape[0], ))
    preds = best_model.predict(X_test)

    # append the predictions
    predictions.append(pd.Series(index=y_test.index, data=preds))

    # score
    scores.append(mean_squared_error(y_test, preds))

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
strat_rets = process_jobs(dates, feats, signal_col='rf_bin_bm')
get_returns_breakout(strat_rets.fillna(0.0).to_frame('rf_benchmark'))




    




