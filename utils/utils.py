import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pandas.tseries.offsets import BDay
import empyrical as ep
from joblib import Parallel, delayed    
import statsmodels.api as sml
from pathlib import Path


def getPortVol(weights, cov, ann_factor=252):
        if ann_factor is None:
            std = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        else:
            std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(ann_factor)
        return std


def get_ret_single_date(data:pd.DataFrame, date:str, signal_col:str, fwd_ret_col='fwd_ret1d',
                        scale_pf_vol_trgt=True, lookback=252, risk_trgt=.15)->pd.Series:
    """
    Calculate the return for a single date based on the given data and parameters.

    Args:
        data (pd.DataFrame): The input data containing the necessary columns.
        date (str): The date for which to calculate the return.
        signal_col (str): The column name for the signal values.
        fwd_ret_col (str, optional): The column name for the forward return values. Defaults to 'fwd_ret1d'.
        scale_pf_vol_trgt (bool, optional): Whether to scale the portfolio volatility to the risk target. Defaults to True.
        lookback (int, optional): The number of days to consider for the lookback period. Defaults to 252.
        risk_trgt (float, optional): The target risk for the portfolio. Defaults to 0.15.

    Returns:
        pd.Series: A series containing the calculated return for the given date.
    """
    
    # output
    out = pd.Series(dtype=np.float64)

    try:
        # copy in the data
        data = data.copy(deep=True)
        data = data.loc[data.index.get_level_values('date')>='1990-01-03']
        data.dropna(subset=[signal_col], inplace=True) 

        one_date = data.loc[data.index.get_level_values('date')==date]
        past_returns = data[['1d_ret']].unstack()
        past_returns.columns = past_returns.columns.droplevel(0)
        past_returns.index = pd.to_datetime(past_returns.index)
        past_returns = past_returns.loc[pd.to_datetime(date) - BDay(lookback): pd.to_datetime(date)]

        del data

        wts = (one_date[signal_col] * (risk_trgt/(one_date['rVol'] * np.sqrt(252))))/one_date['rVol'].count()
        futures = wts.droplevel(0).index.to_list()
        past_returns = past_returns[futures].fillna(0.0)
        
        # from the vols compute a covariance matrix
        cov = past_returns.ewm(span=60).cov()
        cov = cov.loc[cov.index.get_level_values('date') == date].values
        
        # compute the total portfolio standard deviation using the covariance matrix and weights
        total_pf_vol = getPortVol(wts, cov, ann_factor=252)
        
        # scale the total pf volatility to a risk target
        pf_risk_scaler = risk_trgt/total_pf_vol

        # finally we can compute the return for t+1 using the signals and weights at t0
        if scale_pf_vol_trgt:
            wts *= pf_risk_scaler
            ret = wts @ one_date[fwd_ret_col].T
        else:
            ret = wts @ one_date[fwd_ret_col].T

        # append
        out.loc[date] = ret
        return out
    except Exception as e:
        print(f'Failed On: {date}')
        out.loc[date] = np.nan
        return out


def process_jobs(dates, data, signal_col):
    # NOTE with roughly 8k daily observations this takes 20 minutes on cores=24 for a single strategy back-test
    results = Parallel(n_jobs=-1, verbose=True)(delayed(get_ret_single_date)(data, date, signal_col) for date in dates)
    return pd.concat(results, axis=0).sort_index()


def get_returns_breakout(strats: pd.DataFrame):
    """
    tuple of (strat name, series of returns)
    """

    ret_breakout=pd.DataFrame(columns=['Annual_Return', 'Annual_Volatility',
                                       'DD', 'MDD', 'Sharpe', 'Sortino',
                                       'Calmar', 'ppct_postive_rets'])
    for strat in strats.columns:
        _strat = strats[strat]
        ret_breakout.loc[strat, 'Annual_Return'] = ep.annual_return(_strat)
        ret_breakout.loc[strat, 'Annual_Volatility'] = ep.annual_volatility(_strat)
        ret_breakout.loc[strat, 'DD'] = ep.downside_risk(_strat)
        ret_breakout.loc[strat, 'MDD'] = ep.max_drawdown(_strat)
        ret_breakout.loc[strat, 'Sharpe'] = ep.sharpe_ratio(_strat)
        ret_breakout.loc[strat, 'Sortino'] = ep.sortino_ratio(_strat)
        ret_breakout.loc[strat, 'Calmar'] = ep.calmar_ratio(_strat)
        ret_breakout.loc[strat, 'ppct_postive_rets'] = _strat[_strat>0].shape[0]/_strat.shape[0]
       
    return ret_breakout


def build_features(data):
    """
    builds the momentum features mentioned in the paper
    """
    # make copy
    data = data.copy()
    
    # ewm realized volatility a rough forecast of t+1
    data['rVol'] = data.groupby(by='future')[['ret']].pct_change().ewm(span=60).std()

    # trailing returns
    data['1d_ret'] = data.groupby(by='future')['ret'].pct_change(1)
    data['1wk_ret'] = data.groupby(by='future')['ret'].pct_change(5)
    data['1m_ret'] = data.groupby(by='future')['ret'].pct_change(20)
    data['1Q_ret'] = data.groupby(by='future')['ret'].pct_change(60)
    data['6M_ret'] = data.groupby(by='future')['ret'].pct_change(124)
    data['12M_ret'] = data.groupby(by='future')['ret'].pct_change(252)

    # build risk adjusted features
    data['feature_1d_ra'] = data['1d_ret']/data['rVol']
    data['feature_1wk_ra'] = data['1wk_ret']/data['rVol'] * np.sqrt(5)
    data['feature_1m_ra'] = data['1m_ret']/data['rVol'] * np.sqrt(20)
    data['feature_1Q_ra'] = data['1Q_ret']/data['rVol'] * np.sqrt(60)
    data['feature_6M_ra'] = data['6M_ret']/data['rVol'] * np.sqrt(124)
    data['feature_12M_ra'] = data['12M_ret']/data['rVol'] * np.sqrt(252)

    # build moving-average convergence divergence features
    data['feature_MACD_short'] = (data.groupby(by='future')['ret'].ewm(span=8).mean() - data.groupby(by='future')['ret'].ewm(span=24).mean()).droplevel(0)/data.groupby(by='future')['ret'].ewm(span=63).std().droplevel(0)
    data['feature_MACD_medium'] = (data.groupby(by='future')['ret'].ewm(span=16).mean() - data.groupby(by='future')['ret'].ewm(span=48).mean()).droplevel(0)/data.groupby(by='future')['ret'].ewm(span=63).std().droplevel(0)
    data['feature_MACD_long'] = (data.groupby(by='future')['ret'].ewm(span=32).mean() - data.groupby(by='future')['ret'].ewm(span=96).mean()).droplevel(0)/data.groupby(by='future')['ret'].ewm(span=63).std().droplevel(0)

    # build as macd index
    data['feature_MACD_index'] = data[['feature_MACD_short', 'feature_MACD_medium', 'feature_MACD_long']].mean(axis=1)
    data['feature_MACD_index'] = data.groupby(by='future')['feature_MACD_index']/data.groupby(by='future')['feature_MACD_index'].ewm(span=252).std()
    
    # now for new features
    data['NEW_feature_skew6m'] = data.groupby(by='future')['ret'].pct_change(1).rolling(124).skew()
    data['NEW_feature_skew12m'] = data.groupby(by='future')['ret'].pct_change(1).rolling(252).skew()
    data['NEW_feature_kurt6m'] = data.groupby(by='future')['ret'].pct_change(1).rolling(124).kurt()
    data['NEW_feature_kurt12m'] = data.groupby(by='future')['ret'].pct_change(1).rolling(252).kurt()

    # linear trend estimators
    data['NEW_feature_tval3M'] = data.groupby(by='future')['ret'].rolling(60).apply(lambda x: tVarLinR(x)).droplevel(0)
    data['NEW_feature_tval6M'] = data.groupby(by='future')['ret'].rolling(124).apply(lambda x: tVarLinR(x)).droplevel(0)
    data['NEW_feature_tval12M'] = data.groupby(by='future')['ret'].rolling(252).apply(lambda x: tVarLinR(x)).droplevel(0)

    # also build the target - target is +1D risk adjusted return
    data['fwd_ret1d'] = data.groupby(by='future')['1d_ret'].shift(-1)
    data['target'] = data['fwd_ret1d']/data['rVol']
    data['targetBin'] = np.sign(data['target'])

    return data


# code to build features
def tVarLinR(close: pd.Series) -> float:
    # a series of closing prices
    x = np.ones((close.shape[0], 2))
    x[:, 1] = np.arange(close.shape[0])
    ols = sml.OLS(np.log(close), x).fit()
    return ols.tvalues[1]

# NOTE - Benchmarks , creates the signal column
def linear_trend_bm(feats)-> pd.DataFrame:
    """
    creates one of our benchmark stratgies, trend following with linear regression 
    """
    # create a blended index
    feats['lin_trend_benchmark'] = feats[['NEW_feature_tval12M']].mean(axis=1)
    feats['lin_trend_benchmark'] = np.sign(feats['lin_trend_benchmark'])
    return feats

def tsmom_bm(feats)-> pd.DataFrame:
    """
    creates one of our benchmark stratgies, trend following with time-series momemtum
    """
    feats['tsmom_benchmark'] = np.sign(feats['12M_ret'])
    return feats

def long_only_bm(feats):
    """"
    long only benchmark just for reference
    """
    feats['long_only'] = 1.
    return feats


def macd_combined_bm(feats, exp=True):
    """
    macd signal benchmark
    """
    if exp:
        # per Baz et al. 2015: Equation (4)
        feats['feature_MACD_index'] = (feats['feature_MACD_index'] * (np.exp(-feats['feature_MACD_index']**2/4)))/.89
    else:
        feats['feature_MACD_index'] = np.sign(feats['feature_MACD_index'])
   
    return feats


def plot_strats(strats:pd.DataFrame, log_scale=True):
    # TODO - need to be checked log-scaling looked weird should maybe start with the growth of 10k and not 1?
    cumrets = (1+strats).cumprod()
    if log_scale:
        cumrets = np.log(cumrets)

    cumrets.plot(figsize=(12, 8), title='Strategy Performance', grid=True)
    plt.legend(loc='upper left',bbox_to_anchor=(1.0, 1.0))
    plt.show()
    plt.savefig('strat_perf.png', dpi=300, bbox_inches='tight')


def load_features()-> pd.DataFrame:
    """
    Creates the features if they don't already exist otherwise reads them from disk
    """

    root = Path(__file__).parents[1].__str__()

    try:
        feats = pd.read_parquet(root+'\\'+'features.parquet')
        return feats
    except Exception as e:
        tr_index = pd.read_parquet(root+'\\'+'future_total_return_index.parquet')
        feats = build_features(tr_index)

        # save out now 
        feats.to_parquet(root+'\\''features.parquet')
        return feats


# NOTE - functions we will use for cross-validation 
def cv_date_splitter(dates: list, split_length: int=252 * 5) -> list:
    """
    returns time points for expanding window cross-valiation (start, end, test)
    """
    out = []
    start, end = None, None
    num_splits = len(dates)//split_length
    print(num_splits)

    for k, split in enumerate(range(num_splits)):
        if k==0:
            start = dates[0]
            end = dates[split_length]
            out.append((start, end, dates[split_length *(k+2)]))

        elif k>0 and k<num_splits-1:
            start = dates[0]
            end = dates[split_length*(k+1) + 1]
            out.append((start, end, dates[split_length *(k+2)]))

        elif k==num_splits-1:
            start = dates[0]
            end_first = dates[split_length*(k+1) + 1]
            out.append((start, end_first, dates[-1]))
    return out


def get_cv_splits(feats: pd.DataFrame, split_length: int=252*5):
    """
    yields train, test splits that can be used in the evaulation loop
    """
    # get the splits
    splits = cv_date_splitter(feats.index.get_level_values('date').unique(), split_length=split_length)

    for split in splits:
        train = feats.loc[(feats.index.get_level_values('date')>=split[0]) & (feats.index.get_level_values('date')<=split[1])]
        test = feats.loc[(feats.index.get_level_values('date')>split[1]) & (feats.index.get_level_values('date')<=split[2])]
        yield train, test 
       
