import pandas as pd
import numpy as np
from sklearn.ensemble import  RandomForestRegressor

from utils.utils import get_cv_splits, load_features

from sklearn.metrics import mean_squared_error
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
                               n_jobs=-1)

# NOTE - we can either frame as a regression or classification problem in this example we use regression
predictions = []
scores = []
for train, test in tqdm(get_cv_splits(X)):
    # break out X and y train, test
    X_train, y_train = train[features], train[target] 
    X_test, y_test = test[features], test[target]

    # fit the model
    baseRF.fit(X_train, y_train.values.reshape(y_train.shape[0], ))
    preds = baseRF.predict(X_test)

    # append the predictions
    predictions.append(pd.Series(index=y_test.index, data=preds))

    # score
    scores.append(mean_squared_error(y_test, preds))

# print the scores, concat the predictions , to feed into our back-test code
predictions = pd.concat(predictions).sort_index()
scores = np.array(scores)

predictions = predictions.to_frame('rf_bm')
predictions['rf_bin_bm'] = np.siqn(predictions['rf_bm'])



    




