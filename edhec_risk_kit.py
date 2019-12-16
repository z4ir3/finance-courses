import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
from numpy.linalg import inv

# ---------------------------------------------------------------------------------
# Load and format data files
# ---------------------------------------------------------------------------------
def path_to_data_folder():
    return "/Users/mariacristinasampaolo/Documents/python/git-tracked/finance-courses/data/" 

def get_ffme_returns():
    '''
    Returns the French-Fama dataset for the returns of the bottom and top 
    deciles (Low 10 (Small Caps) and Hi 10 (Large Caps)) of US stocks
    '''
    filepath = path_to_data_folder() + "Portfolios_Formed_on_ME_monthly_EW.csv"
    rets = pd.read_csv(filepath, index_col=0, parse_dates=True, na_values=-99.99)
    rets = rets[["Lo 10", "Hi 10"]] / 100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period("M") #.to_period("M") forces the index to be monthly period...
    return rets 
   
def get_hfi_returns():
    '''
    Returns the EDHEC Hedge Funds Index returns
    '''
    filepath = path_to_data_folder() + "edhec-hedgefundindices.csv"
    hfi = pd.read_csv(filepath, index_col=0, parse_dates=True, na_values=-99.99) / 100.0
    # the index is already of type datetime
    return hfi 

def get_brka_rets(monthly=False):
    '''
    Load and format Berkshire Hathaway's returns from 1990-01 to 2018-12.
    Default data are daily returns. 
    If monthly=True, then monthly data are returned. Here, the method used 
    the .resample method which allows to run an aggregation function on each  
    group of returns of the daily time series.
    '''
    filepath = path_to_data_folder() + "brka_d_ret.csv"
    rets = pd.read_csv(filepath, index_col=0, parse_dates=True)
    if monthly:
        rets = rets.resample("M").apply( compound ).to_period("M")
    return rets

def get_fff_returns():
    '''
    Load the Fama-French Research Factors Monthly Dataset.
    Factors returned are those of the Fama-French model:
    - Excess return of the market, i.e., Market minus Risk-Free Rate,
    - Small (size) Minus Big (size) SMB,
    - High (B/P ratio) Minus Low (B/P ratio) HML, 
    - and the Risk Free Rate 
    '''
    filepath = path_to_data_folder() + "F-F_Research_Data_Factors_m.csv"
    fff = pd.read_csv(filepath, index_col=0, parse_dates=True, na_values=-99.99) / 100
    fff.index = pd.to_datetime(fff.index, format="%Y%m").to_period("M")
    return fff 

def get_ind_file(filetype="rets", nind=30, ew=False):
    '''
    Load and format the Kenneth French Industry Portfolios files.
    - filetype: can be "rets", "nfirms", "size"
    - nind: can be 30 or 49
    - ew: if True, equally weighted portfolio dataset are loaded.
      Also, it has a role only when filetype="rets".
    '''
    if nind!=30 and nind!=49:
        raise ValueError("Expected either 30 or 49 number of industries")
    if filetype is "rets":
        portfolio_w = "ew" if ew==True else "vw" 
        name = "{}_rets" .format( portfolio_w )
        divisor = 100.0
    elif filetype is "nfirms":
        name = "nfirms"
        divisor = 1
    elif filetype is "size":
        name = "size"
        divisor = 1
    else:
        raise ValueError("filetype must be one of: rets, nfirms, size")
    filepath = path_to_data_folder() + "ind{}_m_{}.csv" .format(nind, name)
    ind = pd.read_csv(filepath, index_col=0, parse_dates=True) / divisor
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_market_caps(nind=30, weights=False):
    '''
    Load the industry portfolio dataset and returns single industries market caps.
    If weights=True, it returns single industries market cap-weights as a percentage of
    of the total market cap.
    '''
    ind_nfirms = get_ind_file(filetype="nfirms", nind=nind)
    ind_size   = get_ind_file(filetype="size", nind=nind)
    # compute the market capitalization of each industry sector
    ind_caps   = ind_nfirms * ind_size
    if weights:
        # compute the total market capitalization
        total_cap = ind_caps.sum(axis=1)
        # compute single market capitalizations as a percentage of the total market cap
        ind_cap_weight = ind_caps.divide(total_cap, axis=0)
        return ind_cap_weight
    else:
        return ind_caps 
    
def get_total_market_index_returns(nind=30):
    '''
    Computes the returns of a cap-weighted total market index from Kenneth French Industry portfolios
    '''  
    # load the right returns 
    ind_rets = get_ind_file(filetype="rets", nind=nind) 
    # load the cap-weights of each industry 
    ind_cap_weight = get_ind_market_caps(nind=nind, weights=True)
    # total market returns         
    total_market_return = (ind_cap_weight * ind_rets).sum(axis=1)
    return total_market_return

def get_total_market_index(nind=30, capital=1000):
    '''
    Return the cap-weighted total market index from Kenneth French Industry portfolios
    ''' 
    total_market_return = get_total_market_index_returns(nind=nind)
    total_market_index  = capital * (1 + total_market_return).cumprod()
    return total_market_index

# ---------------------------------------------------------------------------------
# Return Analysis and general statistics
# ---------------------------------------------------------------------------------
def terminal_wealth(s):
    '''
    Computes the terminal wealth of a sequence of return, which is, in other words, 
    the final compounded return. 
    The input s is expected to be either a pd.DataFrame or a pd.Series
    '''
    if not isinstance(s, (pd.DataFrame, pd.Series)):
        raise ValueError("Expected either a pd.DataFrame or pd.Series")
    return (1 + s).prod()

def compound(s):
    '''
    Single compound rule for a pd.Dataframe or pd.Series of returns. 
    The method returns a single number - using prod(). 
    See also the TERMINAL_WEALTH method.
    '''
    if not isinstance(s, (pd.DataFrame, pd.Series)):
        raise ValueError("Expected either a pd.DataFrame or pd.Series")
    return (1 + s).prod() - 1
    # Note that this is equivalent to (but slower than)
    # return np.expm1( np.logp1(s).sum() )
    
def compound_returns(s, start=100):
    '''
    Compound a pd.Dataframe or pd.Series of returns from an initial default value equal to 100.
    In the former case, the method compounds the returns for every column (Series) by using pd.aggregate. 
    The method returns a pd.Dataframe or pd.Series - using cumprod(). 
    See also the COMPOUND method.
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( compound_returns, start=start )
    elif isinstance(s, pd.Series):
        return start * (1 + s).cumprod()
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
        
def compute_returns(s):
    '''
    Computes the returns (percentage change) of a Dataframe of Series. 
    In the former case, it computes the returns for every column (Series) by using pd.aggregate
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( compute_returns )
    elif isinstance(s, pd.Series):
        return s / s.shift(1) - 1
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
        
def compute_logreturns(s):
    '''
    Computes the log-returns of a Dataframe of Series. 
    In the former case, it computes the returns for every column (Series) by using pd.aggregate
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( compute_logreturns )
    elif isinstance(s, pd.Series):
        return np.log( s / s.shift(1) )
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
    
def drawdown(rets: pd.Series, start=1000):
    '''
    Compute the drawdowns of an input pd.Series of returns. 
    The method returns a dataframe containing: 
    1. the associated wealth index (for an hypothetical starting investment of $1000) 
    2. all previous peaks 
    3. the drawdowns
    '''
    wealth_index   = compound_returns(rets, start=start)
    previous_peaks = wealth_index.cummax()
    drawdowns      = (wealth_index - previous_peaks ) / previous_peaks
    df = pd.DataFrame({"Wealth": wealth_index, "Peaks": previous_peaks, "Drawdown": drawdowns} )
    return df

def skewness(s):
    '''
    Computes the Skewness of the input Series or Dataframe.
    There is also the function scipy.stats.skew().
    '''
    return ( ((s - s.mean()) / s.std(ddof=0))**3 ).mean()

def kurtosis(s):
    '''
    Computes the Kurtosis of the input Series or Dataframe.
    There is also the function scipy.stats.kurtosis() which, however, 
    computes the "Excess Kurtosis", i.e., Kurtosis minus 3
    '''
    return ( ((s - s.mean()) / s.std(ddof=0))**4 ).mean()

def exkurtosis(s):
    '''
    Returns the Excess Kurtosis, i.e., Kurtosis minus 3
    '''
    return kurtosis(s) - 3

def is_normal(s, level=0.01):
    '''
    Jarque-Bera test to see if a series (of returns) is normally distributed.
    Returns True or False according to whether the p-value is larger 
    than the default level=0.01.
    '''
    statistic, pvalue = scipy.stats.jarque_bera( s )
    return pvalue > level

def semivolatility(s):
    '''
    Returns the semivolatility of a series, i.e., the volatility of
    negative returns
    '''
    return s[s<0].std(ddof=0) 

def var_historic(s, level=0.05):
    '''
    Returns the (1-level)% VaR using historical method. 
    By default it computes the 95% VaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the VaR for every column (Series) by using pd.aggregate
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( var_historic, level=level )
    elif isinstance(s, pd.Series):
        return - np.percentile(s, level*100)
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
        
def var_gaussian(s, level=0.05, cf=False):
    '''
    Returns the (1-level)% VaR using the parametric Gaussian method. 
    By default it computes the 95% VaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
    The variable "cf" stands for Cornish-Fisher. If True, the method computes the 
    modified VaR using the Cornish-Fisher expansion of quantiles.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the VaR for every column (Series).
    '''
    # alpha-quantile of Gaussian distribution 
    za = scipy.stats.norm.ppf(level,0,1) 
    if cf:
        S = skewness(s)
        K = kurtosis(s)
        za = za + (za**2 - 1)*S/6 + (za**3 - 3*za)*(K-3)/24 - (2*za**3 - 5*za)*(S**2)/36    
    return -( s.mean() + za * s.std(ddof=0) )

def cvar_historic(s, level=0.05):
    '''
    Computes the (1-level)% Conditional VaR (based on historical method).
    By default it computes the 95% CVaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the VaR for every column (Series).
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( cvar_historic, level=level )
    elif isinstance(s, pd.Series):
        # find the returns which are less than (the historic) VaR
        mask = s < -var_historic(s, level=level)
        # and of them, take the mean 
        return -s[mask].mean()
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")

def annualize_rets(s, periods_per_year):
    '''
    Computes the return per year, or, annualized return.
    The variable periods_per_year can be, e.g., 12, 52, 252, in 
    case of monthly, weekly, and daily data.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the annualized return for every column (Series) by using pd.aggregate
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( annualize_rets, periods_per_year=periods_per_year )
    elif isinstance(s, pd.Series):
        growth = (1 + s).prod()
        n_period_growth = s.shape[0]
        return growth**(periods_per_year/n_period_growth) - 1

def annualize_vol(s, periods_per_year):
    '''
    Computes the volatility per year, or, annualized volatility.
    The variable periods_per_year can be, e.g., 12, 52, 252, in 
    case of monthly, weekly, and daily data.
    The method takes in input either a DataFrame, a Series, a list or a single number. 
    In the former case, it computes the annualized volatility of every column 
    (Series) by using pd.aggregate. In the latter case, s is a volatility 
    computed beforehand, hence only annulization is done
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(annualize_vol, periods_per_year=periods_per_year )
    elif isinstance(s, pd.Series):
        return s.std() * (periods_per_year)**(0.5)
    elif isinstance(s, list):
        return np.std(s) * (periods_per_year)**(0.5)
    elif isinstance(s, (int,float)):
        return s * (periods_per_year)**(0.5)

def sharpe_ratio(s, risk_free_rate, periods_per_year, v=None):
    '''
    Computes the annualized sharpe ratio. 
    The variable periods_per_year can be, e.g., 12, 52, 252, in case of yearly, weekly, and daily data.
    The variable risk_free_rate is the annual one.
    The method takes in input either a DataFrame, a Series or a single number. 
    In the former case, it computes the annualized sharpe ratio of every column (Series) by using pd.aggregate. 
    In the latter case, s is the (allready annualized) return and v is the (already annualized) volatility 
    computed beforehand, for example, in case of a portfolio.
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( sharpe_ratio, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year, v=None)
    elif isinstance(s, pd.Series):
        # convert the annual risk free rate to the period assuming that:
        # RFR_year = (1+RFR_period)^{periods_per_year} - 1. Hence:
        rf_to_period = (1 + risk_free_rate)**(1/periods_per_year) - 1        
        excess_return = s - rf_to_period
        # now, annualize the excess return
        ann_ex_rets = annualize_rets(excess_return, periods_per_year)
        # compute annualized volatility
        ann_vol = annualize_vol(s, periods_per_year)
        return ann_ex_rets / ann_vol
    elif isinstance(s, (int,float)) and v is not None:
        # Portfolio case: s is supposed to be the single (already annnualized) 
        # return of the portfolio and v to be the single (already annualized) volatility. 
        return (s - risk_free_rate) / v

# ---------------------------------------------------------------------------------
# Modern Portfolio Theory 
# ---------------------------------------------------------------------------------
def portfolio_return(weights, vec_returns):
    '''
    Computes the return of a portfolio. 
    It takes in input a row vector of weights (list of np.array) 
    and a column vector (or pd.Series) of returns
    '''
    return np.dot(weights, vec_returns)
    
def portfolio_volatility(weights, cov_rets):
    '''
    Computes the volatility of a portfolio. 
    It takes in input a vector of weights (np.array or pd.Series) 
    and the covariance matrix of the portfolio asset returns
    '''
    return ( np.dot(weights.T, np.dot(cov_rets, weights)) )**(0.5) 

def efficient_frontier(n_portfolios, rets, covmat, periods_per_year, risk_free_rate=0.0, 
                       iplot=False, hsr=False, cml=False, mvp=False, ewp=False):
    '''
    Returns (and plots) the efficient frontiers for a portfolio of rets.shape[1] assets. 
    The method returns a dataframe containing the volatilities, returns, sharpe ratios and weights 
    of the portfolios as well as a plot of the efficient frontier in case iplot=True. 
    Other inputs are:
        hsr: if true the method plots the highest return portfolio,
        cml: if True the method plots the capital market line;
        mvp: if True the method plots the minimum volatility portfolio;
        ewp: if True the method plots the equally weigthed portfolio. 
    The variable periods_per_year can be, e.g., 12, 52, 252, in case of monthly, weekly, and daily data.
    '''   
    
    def append_row_df(df,vol,ret,spr,weights):
        temp_df = list(df.values)
        temp_df.append( [vol, ret, spr,] + [w for w in weights] )
        return pd.DataFrame(temp_df)
        
    ann_rets = annualize_rets(rets, periods_per_year)
    
    # generates optimal weights of porfolios lying of the efficient frontiers
    weights = optimal_weights(n_portfolios, ann_rets, covmat, periods_per_year) 
    # in alternative, if only the portfolio consists of only two assets, the weights can be: 
    #weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_portfolios)]

    # portfolio returns
    portfolio_ret = [portfolio_return(w, ann_rets) for w in weights]
    
    # portfolio volatility
    vols          = [portfolio_volatility(w, covmat) for w in weights] 
    portfolio_vol = [annualize_vol(v, periods_per_year) for v in vols]
    
    # portfolio sharpe ratio
    portfolio_spr = [sharpe_ratio(r, risk_free_rate, periods_per_year, v=v) for r,v in zip(portfolio_ret,portfolio_vol)]
    
    df = pd.DataFrame({"volatility": portfolio_vol,
                       "return": portfolio_ret,
                       "sharpe ratio": portfolio_spr})
    df = pd.concat([df, pd.DataFrame(weights)],axis=1)
    
    if iplot:
        ax = df.plot.line(x="volatility", y="return", style="--", color="coral", grid=True, label="Efficient frontier", figsize=(8,4))
        if hsr or cml:
            w   = maximize_shape_ratio(ann_rets, covmat, risk_free_rate, periods_per_year)
            ret = portfolio_return(w, ann_rets)
            vol = annualize_vol( portfolio_volatility(w,covmat), periods_per_year)
            spr = sharpe_ratio(ret, risk_free_rate, periods_per_year, v=vol)
            df  = append_row_df(df,vol,ret,spr,w)
            if cml:
                # Draw the CML: the endpoints of the CML are [0,risk_free_rate] and [port_vol,port_ret]
                ax.plot([0, vol], [risk_free_rate, ret], color="g", linestyle="-.", label="CML")
                ax.set_xlim(left=0)
                ax.legend()
            if hsr:
                # Plot the highest sharpe ratio portfolio
                ax.scatter([vol], [ret], marker="o", color="g", label="MSR portfolio")
                ax.legend()
        if mvp:
            # Plot the global minimum portfolio:
            w   = minimize_volatility(ann_rets, covmat)
            ret = portfolio_return(w, ann_rets)
            vol = annualize_vol( portfolio_volatility(w,covmat), periods_per_year)
            spr = sharpe_ratio(ret, risk_free_rate, periods_per_year, v=vol)
            df  = append_row_df(df,vol,ret,spr,w)
            ax.scatter([vol], [ret], color="midnightblue", marker="o", label="GMV portfolio")
            ax.legend()  
        if ewp:
            # Plot the equally weighted portfolio:
            w   = np.repeat(1/ann_rets.shape[0], ann_rets.shape[0])
            ret = portfolio_return(w, ann_rets)
            vol = annualize_vol( portfolio_volatility(w,covmat), periods_per_year)
            spr = sharpe_ratio(ret, risk_free_rate, periods_per_year, v=vol)
            df  = append_row_df(df,vol,ret,spr,w)
            ax.scatter([vol], [ret], color="goldenrod", marker="o", label="EW portfolio")
            ax.legend()
        return df, ax
    else: 
        return df
    
def summary_stats(s, risk_free_rate=0.03, periods_per_year=12, var_level=0.05):
    '''
    Returns a dataframe containing annualized returns, annualized volatility, sharpe ratio, 
    skewness, kurtosis, historic VaR, Cornish-Fisher VaR, and Max Drawdown
    '''
    if isinstance(s, pd.Series):
        stats = {
            "Ann. return"  : annualize_rets(s, periods_per_year=periods_per_year),
            "Ann. vol"     : annualize_vol(s, periods_per_year=periods_per_year),
            "Sharpe ratio" : sharpe_ratio(s, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
            "Skewness"     : skewness(s),
            "Kurtosis"     : kurtosis(s),
            "Historic CVar": cvar_historic(s, level=var_level),
            "C-F Var"      : var_gaussian(s, level=var_level, cf=True),
            "Max drawdown" : drawdown(s)["Drawdown"].min()
        }
        return pd.DataFrame(stats, index=["0"])
    
    elif isinstance(s, pd.DataFrame):        
        stats = {
            "Ann. return"  : s.aggregate( annualize_rets, periods_per_year=periods_per_year ),
            "Ann. vol"     : s.aggregate( annualize_vol,  periods_per_year=periods_per_year ),
            "Sharpe ratio" : s.aggregate( sharpe_ratio, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year ),
            "Skewness"     : s.aggregate( skewness ),
            "Kurtosis"     : s.aggregate( kurtosis ),
            "Historic CVar": s.aggregate( cvar_historic, level=var_level ),
            "C-F Var"      : s.aggregate( var_gaussian, level=var_level, cf=True ),
            "Max Drawdown" : s.aggregate( lambda r: drawdown(r)["Drawdown"].min() )
        } 
        return pd.DataFrame(stats)
    
def summary_stats_terminal(rets, floor=0.8, periods_per_year=2, name="Stats", target=np.inf):
    '''
    Return a dataframe of statistics for a given input pd.DataFrame of asset returns. 
    Statistics computed are:
    - the mean annualized return
    - the mean terminal wealth (compounded return)
    - the mean terminal wealth volatility
    - the probability that an input floor is breached by terminal wealths
    - the expected shortfall of those terminal wealths breaching the input floor 
    '''    
    # terminal wealths over scenarios, i.e., compounded returns
    terminal_wlt = terminal_wealth(rets)
    
    # boolean vector of terminal wealths going below the floor 
    floor_breach = terminal_wlt < floor

    stats = pd.DataFrame.from_dict({
        "Mean ann. ret.":  annualize_rets(rets, periods_per_year=periods_per_year).mean(),              # mean annualized returns over scenarios
        "Mean wealth":     terminal_wlt.mean(),                                                         # terminal wealths mean 
        "Mean wealth std": terminal_wlt.std(),                                                          # terminal wealths volatility
        "Prob breach":     floor_breach.mean() if floor_breach.sum() > 0 else 0,                        # probability of breaching the floor
        "Exp shortfall":   (floor - terminal_wlt[floor_breach]).mean() if floor_breach.sum() > 0 else 0 # expected shortfall if floor is reached  
    }, orient="index", columns=[name])
    return stats
     
def optimal_weights(n_points, rets, covmatrix, periods_per_year):
    '''
    Returns a set of n_points optimal weights corresponding to portfolios (of the efficient frontier) 
    with minimum volatility constructed by fixing n_points target returns. 
    The weights are obtained by solving the minimization problem for the volatility. 
    '''
    target_rets = np.linspace(rets.min(), rets.max(), n_points)    
    weights = [minimize_volatility(rets, covmatrix, target) for target in target_rets]
    return weights

def minimize_volatility(rets, covmatrix, target_return=None):
    '''
    Returns the optimal weights of the minimum volatility portfolio on the effient frontier. 
    If target_return is not None, then the weights correspond to the minimum volatility portfolio 
    having a fixed target return. 
    The method uses the scipy minimize optimizer which solves the minimization problem 
    for the volatility of the portfolio
    '''
    n_assets = rets.shape[0]    
    # initial guess weights
    init_guess = np.repeat(1/n_assets, n_assets)
    weights_constraint = {
        "type": "eq",
        "fun": lambda w: 1.0 - np.sum(w)  
    }
    if target_return is not None:
        return_constraint = {
            "type": "eq",
            "args": (rets,),
            "fun": lambda w, r: target_return - portfolio_return(w, r)
        }
        constr = (return_constraint, weights_constraint)
    else:
        constr = weights_constraint
        
    result = minimize(portfolio_volatility, 
                      init_guess,
                      args = (covmatrix,),
                      method = "SLSQP",
                      options = {"disp": False},
                      constraints = constr,
                      bounds = ((0.0,1.0),)*n_assets ) # bounds of each individual weight, i.e., w between 0 and 1
    return result.x

def minimize_volatility_2(rets, covmatrix, target_return=None, weights_norm_const=True, weights_bound_const=True):
    '''
    Returns the optimal weights of the minimum volatility portfolio.
    If target_return is not None, then the weights correspond to the minimum volatility portfolio 
    having a fixed target return (such portfolio will be on the efficient frontier).
    The variables weights_norm_const and weights_bound_const impose two more conditions, the firt one on 
    weight that sum to 1, and the latter on the weights which have to be between zero and 1
    The method uses the scipy minimize optimizer which solves the minimization problem 
    for the volatility of the portfolio
    '''
    n_assets = rets.shape[0]    
    
    # initial guess weights
    init_guess = np.repeat(1/n_assets, n_assets)
    
    if weights_bound_const:
        # bounds of the weights (between 0 and 1)
        bounds = ((0.0,1.0),)*n_assets
    else:
        bounds = None
    
    constraints = []
    if weights_norm_const:
        weights_constraint = {
            "type": "eq",
            "fun": lambda w: 1.0 - np.sum(w)  
        }
        constraints.append( weights_constraint )    
    if target_return is not None:
        return_constraint = {
            "type": "eq",
            "args": (rets,),
            "fun": lambda w, r: target_return - portfolio_return(w, r)
        }
        constraints.append( return_constraint )
    
    result = minimize(portfolio_volatility, 
                      init_guess,
                      args = (covmatrix,),
                      method = "SLSQP",
                      options = {"disp": False},
                      constraints = tuple(constraints),
                      bounds = bounds)
    return result.x

def maximize_shape_ratio(rets, covmatrix, risk_free_rate, periods_per_year, target_volatility=None):
    '''
    Returns the optimal weights of the highest sharpe ratio portfolio on the effient frontier. 
    If target_volatility is not None, then the weights correspond to the highest sharpe ratio portfolio 
    having a fixed target volatility. 
    The method uses the scipy minimize optimizer which solves the maximization of the sharpe ratio which 
    is equivalent to minimize the negative sharpe ratio.
    '''
    n_assets   = rets.shape[0] 
    init_guess = np.repeat(1/n_assets, n_assets)
    weights_constraint = {
        "type": "eq",
        "fun": lambda w: 1.0 - np.sum(w)  
    }
    if target_volatility is not None:
        volatility_constraint = {
            "type": "eq",
            "args": (covmatrix, periods_per_year),
            "fun": lambda w, cov, p: target_volatility - annualize_vol(portfolio_volatility(w, cov), p)
        }
        constr = (volatility_constraint, weights_constraint)
    else:
        constr = weights_constraint
        
    def neg_portfolio_sharpe_ratio(weights, rets, covmatrix, risk_free_rate, periods_per_year):
        '''
        Computes the negative annualized sharpe ratio for minimization problem of optimal portfolios.
        The variable periods_per_year can be, e.g., 12, 52, 252, in case of yearly, weekly, and daily data.
        The variable risk_free_rate is the annual one.
        '''
        # annualized portfolio returns
        portfolio_ret = portfolio_return(weights, rets)        
        # annualized portfolio volatility
        portfolio_vol = annualize_vol(portfolio_volatility(weights, covmatrix), periods_per_year)
        return - sharpe_ratio(portfolio_ret, risk_free_rate, periods_per_year, v=portfolio_vol)    
        #i.e., simply returns  -(portfolio_ret - risk_free_rate)/portfolio_vol
        
    result = minimize(neg_portfolio_sharpe_ratio,
                      init_guess,
                      args = (rets, covmatrix, risk_free_rate, periods_per_year),
                      method = "SLSQP",
                      options = {"disp": False},
                      constraints = constr,
                      bounds = ((0.0,1.0),)*n_assets)
    return result.x

def weigths_max_sharpe_ratio(covmat, mu_exc, scale=True):
    '''
    Optimal (Tangent/Max Sharpe Ratio) portfolio weights using the Markowitz Optimization Procedure:
    - mu_exc is the vector of Excess expected Returns (has to be a column vector as a pd.Series)
    - covmat is the covariance N x N matrix as a pd.DataFrame
    Look at pag. 188 eq. (5.2.28) of "The econometrics of financial markets", by Campbell, Lo, Mackinlay.
    '''
    w = inverse_df(covmat).dot(mu_exc)
    if scale:
        # normalize weigths
        w = w/sum(w) 
    return w
    
# ---------------------------------------------------------------------------------
# CPPI backtest strategy
# ---------------------------------------------------------------------------------
def cppi(risky_rets, safe_rets=None, start_value=1000, floor=0.8, m=3, drawdown=None,
         risk_free_rate=0.03, periods_per_year=12):
    '''
    Run a backtest of the CPPI investment strategy given a set of returns for a risky asset
    Returns, account value history, risk budget history, and risky weight history
    '''
    
    # compute the risky wealth (100% investment in the risky asset)
    risky_wealth = start_value * (1 + risky_rets).cumprod()

    # CPPI parameters
    account_value = start_value
    floor_value   = floor * account_value
    
    # Make the returns a DataFrame
    if isinstance(risky_rets, pd.Series):
        risky_rets = pd.DataFrame(risky_rets, columns="Risky return")
    
    # If returns of safe assets are not available just make artificial ones 
    if safe_rets is None:
        safe_rets = pd.DataFrame().reindex_like(risky_rets)
        safe_rets[:] = risk_free_rate / periods_per_year
    
    # History dataframes
    account_history = pd.DataFrame().reindex_like(risky_rets)
    cushion_history = pd.DataFrame().reindex_like(risky_rets)
    risky_w_history = pd.DataFrame().reindex_like(risky_rets)
    
    # Extra history dataframes in presence of drawdown
    if drawdown is not None:
        peak_history  = pd.DataFrame().reindex_like(risky_rets)
        floor_history = pd.DataFrame().reindex_like(risky_rets)
        peak = start_value
        # define the multiplier 
        m = 1 / drawdown
    
    # For loop over dates 
    for step in range( len(risky_rets.index) ):
        if drawdown is not None:
            # current peak
            peak = np.maximum(peak, account_value)
            # current floor value
            floor_value = peak * (1 - drawdown)
            floor_history.iloc[step] = floor_value
        
        # computing the cushion (as a percentage of the current account value)
        cushion = (account_value - floor_value) / account_value
    
        # compute the weight for the allocation on the risky asset
        risky_w = m * cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        # the last two conditions ensure that the risky weight is in [0,1]
    
        # compute the weight for the allocation on the safe asset
        safe_w  = 1 - risky_w

        # compute the value allocation
        risky_allocation = risky_w * account_value
        safe_allocation  = safe_w  * account_value

        # compute the new account value: this is given by the new values from both the risky and the safe assets
        account_value = risky_allocation * (1 + risky_rets.iloc[step] ) + safe_allocation  * (1 + safe_rets.iloc[step]  )

        # save data: current account value, cushions, weights
        account_history.iloc[step] = account_value
        cushion_history.iloc[step] = cushion 
        risky_w_history.iloc[step] = risky_w
    
    # Given the CPPI wealth saved in the account_history, we can get back the CPPI returns
    cppi_rets = ( account_history / account_history.shift(1) - 1 ).dropna()
    
    # Returning results
    backtest_result = {
        "Risky wealth"    : risky_wealth, 
        "CPPI wealth"     : account_history, 
        "CPPI returns"    : cppi_rets, 
        "Cushions"        : cushion_history,
        "Risky allocation": risky_w_history,
        "Safe returns"    : safe_rets
    }
    if drawdown is not None:
        backtest_result.update({
            "Floor value": floor_history,
            "Peaks"      : peak_history,
            "m"          : m
        })

    return backtest_result

# ---------------------------------------------------------------------------------
# Random walks
# ---------------------------------------------------------------------------------
def simulate_gbm_from_returns(n_years=10, n_scenarios=20, mu=0.07, sigma=0.15, periods_per_year=12, start=100.0):
    '''
    Evolution of an initial stock price using Geometric Brownian Model:
        (S_{t+dt} - S_t)/S_t = mu*dt + sigma*sqrt(dt)*xi,
    where xi are normal random variable N(0,1). 
    The equation for percentage returns above is used to generate returns and they are compounded 
    in order to get the prices.    
    Note that default periods_per_year=12 means that the method generates monthly prices (and returns):
    change to 52 or 252 for weekly or daily prices and returns, respectively.
    The method returns a dataframe of prices and the dataframe of returns.
    '''
    dt = 1 / periods_per_year
    n_steps = int(n_years * periods_per_year)
    
    # from GBM equation for percentage returns, returns have mean = mu*dt and std = sigma*sqrt(dt)
    rets = pd.DataFrame( np.random.normal(loc=mu*dt, scale=sigma*(dt)**(0.5), size=(n_steps, n_scenarios)) )
    
    # compute prices by compound the generated returns
    prices = compound_returns(rets, start=start)
    prices = insert_first_row_df(prices, start)
    
    return prices, rets

def simulate_gbm_from_prices(n_years=10, n_scenarios=20, mu=0.07, sigma=0.15, periods_per_year=12, start=100.0):
    '''
    Evolution of an initial stock price using Geometric Brownian Model:
        S_t = S_0 exp( (mu-sigma^2/2)*dt + sigma*sqrt(dt)*xi ), 
    where xi are normal random variable N(0,1). 
    The equation for (log-)returns above is used to generate the prices and then log-returns are 
    computed by definition of log(S_{t+dt}/S_t). 
    Note that default periods_per_year=12 means that the method generates monthly prices (and returns):
    change to 52 or 252 for weekly or daily prices and returns, respectively.
    The method returns a dataframe of prices and the dataframe of returns.
    '''
    dt = 1 / periods_per_year
    n_steps = int(n_years * periods_per_year)
    
    # from GBM equation for log-prices:
    prices_dt = np.exp( np.random.normal(loc=(mu - 0.5*sigma**2)*dt, scale=sigma*(dt**(0.5)), size=(n_steps, n_scenarios)) )
    # equivalent (but faster) to: 
    # prices_dt = np.exp( (mu - 0.5*sigma**2)*dt + sigma*np.random.normal(loc=0, scale=(dt)**(0.5), size=(n_steps, n_scenarios)) )    
    prices = start * pd.DataFrame(prices_dt).cumprod()
    prices = insert_first_row_df(prices, start)
    
    # compute log-returns from generated prices
    rets = compute_logreturns(prices).dropna()
    
    return prices, rets

def show_gbm(n_years=10, n_scenarios=10, mu=0.05, sigma=0.15, periods_per_year=12, start=100):
    '''
    Plot the evolution of prices genrated by a GBM. 
    The method simply calls the *simulate_gbm_from_returns* function and plot the genrated prices. 
    This method is designed to be used together with the *interact* method form *ipywidgets*. 
    '''
    prices, rets = simulate_gbm_from_returns(n_years=n_years, n_scenarios=n_scenarios, 
                                             mu=mu, sigma=sigma, periods_per_year=periods_per_year, start=start)
    ax = prices.plot(figsize=(12,5), grid=True, legend=False, color="sandybrown", alpha=0.7, linewidth=2)
    ax.axhline(y=start, ls=":", color="black")
    if periods_per_year == 12:
        xlab = "months"
    elif periods_per_year == 52:
        xlab = "weeks"
    elif periods_per_year == 252:
        xlab = "days"
    ax.set_xlabel(xlab)
    ax.set_ylabel("price")
    ax.set_title("Prices generated by GBM")
    
def show_cppi(n_years=10, n_scenarios=50, m=3, floor=0, mu=0.04, sigma=0.15, 
              risk_free_rate=0.03, periods_per_year=12, start=100, ymax=100):
    '''
    CPPI simulation using Brownian Motion generated returns with mean mu and std sigma. 
    The method will plot the simulated CPPI wealths as well as an histogram of the
    CPPI wealths at the end of the given period (n_year).
    '''
    # generate returs using geometric brownian motions 
    _, risky_rets = simulate_gbm_from_returns(n_years=n_years, n_scenarios=n_scenarios, mu=mu, sigma=sigma, 
                                              periods_per_year=periods_per_year, start=start)
    
    # run the CPPI strategy with fixed floor (i.e., with no drawdown constraint)
    cppiw = cppi(risky_rets, start_value=start, floor=floor, m=m, drawdown=None, 
                 risk_free_rate=risk_free_rate, periods_per_year=periods_per_year )["CPPI wealth"]

    # make sure that start price is included  
    cols = [i for i in range(0,cppiw.shape[1])]
    row = {}
    for col in cols:
        row[col] = start
    cppiw = insert_first_row_df(cppiw, row)
    
    # Plot parameters
    fig, (wealth_ax, hist_ax) = plt.subplots(figsize=(20,7), nrows=1,ncols=2,sharey=True, gridspec_kw={"width_ratios":[3,2]} )
    plt.subplots_adjust(wspace=0.005)    
    simclr   = "sandybrown"
    floorclr = "red"
    startclr = "black"
    ymax = (cppiw.values.max() - start)/100*ymax + start
    
    # Plot the random walks
    cppiw.plot(ax=wealth_ax, grid=True, legend=False, color=simclr, alpha=0.5, linewidth=2)
    wealth_ax.axhline(y=start, ls=":", color=startclr)
    wealth_ax.axhline(y=start*floor, ls=":", color=floorclr, linewidth=2)
    if periods_per_year == 12:
        xlab = "months"
    elif periods_per_year == 52:
        xlab = "weeks"
    elif periods_per_year == 252:
        xlab = "days"
    wealth_ax.set_xlabel(xlab)
    wealth_ax.set_ylim(top=ymax)
    wealth_ax.set_title("CPPI wealths due to brownian motion generated returns", fontsize=14)
    
    # Plot the histogram
    violations_per_scenarios = (cppiw < start*floor).sum() # number of CPPI wealth violations of the floor per each scenario
    total_violations = violations_per_scenarios.sum()      # overall number of CPPI wealth violations during the entire period

    terminal_wealth = cppiw.iloc[-1]                       # CPPI wealth at the end of the period
    tw_mean      = terminal_wealth.mean()
    tw_median    = terminal_wealth.median() 
    failure_mask = np.less(terminal_wealth, start*floor)
    n_failures   = failure_mask.sum()
    p_fail       = n_failures / n_scenarios
    e_shorfall   = np.dot(terminal_wealth - start*floor,failure_mask) / n_failures if n_failures > 0.0 else 0.0 
    
    terminal_wealth.hist(grid=False, ax=hist_ax, bins=50, ec="white", fc=simclr, orientation="horizontal")
    hist_ax.axhline(y=start, ls=":", color=startclr)
    hist_ax.axhline(y=start*floor, ls=":", color=floorclr, linewidth=2)
    hist_ax.axhline(y=tw_mean, ls=":", color=simclr)
    hist_ax.axhline(y=tw_median, ls=":", color=simclr)
    hist_ax.annotate("Mean: ${:.2f}".format(tw_mean), xy=(0.5, 0.9), xycoords="axes fraction", fontsize=15)
    hist_ax.annotate("Median: ${:.2f}".format(tw_mean), xy=(0.5, 0.85), xycoords="axes fraction", fontsize=15)
    if floor > 0.0:
        hist_ax.annotate("Violations (overall): {}".format(total_violations), xy=(0.5, 0.75), xycoords="axes fraction", fontsize=15)
        hist_ax.annotate("Violations (end period): {} ({:.1f}%)".format(n_failures, p_fail*100), xy=(0.5, 0.7), xycoords="axes fraction", fontsize=15)
        hist_ax.annotate("E(shortfall) (end period): ${:.2f}".format(e_shorfall), xy=(0.5, 0.65), xycoords="axes fraction", fontsize=15)
    hist_ax.set_title("Histogram of the CPPI wealth at the end of the period", fontsize=14)

# ---------------------------------------------------------------------------------
# Securities 
# ---------------------------------------------------------------------------------
def discount(t, r):
    '''
    Compute the price of a pure discount bond that pays 1 at time t (year),
    given an interest rate (return) r. That is, considering FV = 1 at time t, 
    want to obtain the PV given r, i.e., PV = FV/(1+r)^t = 1/(1+r)^t.
    Note that t has to be a pd.Series of times.
    ''' 
    if not isinstance(t, pd.Series):
        t = pd.Series(t)
        
    if not isinstance(r, list):
        r = [r]
        
    ds = pd.DataFrame( [1/(1+rate)**(t) for rate in r] ).T
    ds.index = t
    return ds

def present_value(L, r):
    '''
    Computes the (cumulative) present value PV of a DataFrame
    of liabilities L at a given interest rate r. 
    Liabilities L has to be a pd.DataFrame
    '''
    if not isinstance(L, pd.DataFrame):
        raise TypeError("Expected pd.DataFrame")

    dates = pd.Series(L.index)    
    ds = discount(dates, r)  # this is the series of present values of future cashflows
    return (ds * L).sum()
    
def funding_ratio(asset_value, liabilities, r):
    '''
    Computes the funding ratio between the value of holding assets and the present 
    value of the liabilities given an interest rate r (or a list of)
    '''
    return asset_value / present_value(liabilities, r)   
    
def compounding_rate(r, periods_per_year=None):
    '''
    Given a nominal rate r, it returns the continuously compounded rate R = e^r - 1 if periods_per_year is None.
    If periods_per_year is not None, then returns the discrete compounded rate R = (1+r/N)**N-1.
    '''
    if periods_per_year is None:
        return np.exp(r) - 1
    else:
        return (1 + r/periods_per_year)**periods_per_year - 1
    
def compounding_rate_inv(R, periods_per_year=None):
    '''
    Given a compounded rate, it returns the nominal rate from continuously compounding 
    r = log(1+R) if periods_per_year is None.
    If periods_per_year is not None, then returns the nominal rate from discrete 
    compounding r = N*((1+R)^1/N-1).
    '''
    if periods_per_year is None:
        return np.log(1+R)
    else:
        return periods_per_year * ( (1+R)**(1/periods_per_year) - 1 )

def simulate_cir(n_years=10, n_scenarios=10, a=0.05, b=0.03, sigma=0.05, periods_per_year=12, r0=None):
    '''
    Evolution of (instantaneous) interest rates and corresponding zero-coupon bond using the CIR model:
        dr_t = a*(b-r_t) + sigma*sqrt(r_t)*xi,
    where xi are normal random variable N(0,1). 
    The analytical solution for the zero-coupon bond price is also computed.
    The method returns a dataframe of interest rate and zero-coupon bond prices
    '''
    if r0 is None:
        # Assign the long-term mean interest rate as initial rate
        r0 = b
        
    # Compute the price of a ZCB
    def zcbprice(ttm,r,h):
        A = ( ( 2*h*np.exp(0.5*(a+h)*ttm) ) / ( 2*h + (a+h)*(np.exp(h*ttm)-1) ) )**(2*a*b/(sigma**2))
        B = ( 2*(np.exp(h*ttm)-1) ) / ( 2*h + (a+h)*(np.exp(h*ttm)-1) ) 
        return A * np.exp(-B * r)
    
    dt = 1 / periods_per_year
    n_steps = int(n_years * periods_per_year) + 1
    
    # get the nominal (instantaneous) rate 
    r0 = compounding_rate_inv(r0)
    
    # the schock is sqrt(dt)*xi_t, with xi_t being standard normal r.v.
    shock = np.random.normal(loc=0, scale=(dt)**(0.5), size=(n_steps, n_scenarios))
    
    # Rates initialization
    rates = np.zeros_like(shock)
    rates[0] = r0 
    
    # Price initialization and parameters
    zcb_prices = np.zeros_like(shock)
    h = np.sqrt(a**2 + 2*sigma**2)
    zcb_prices[0] = zcbprice(n_years,r0,h)

    for step in range(1,n_steps):
        # previous interest rate
        r_t = rates[step-1]
        
        # Current (updated) interest rate: CIR equation
        rates[step] = r_t + a*(b - r_t) + sigma*np.sqrt(r_t)*shock[step]
        
        # Current (updated) ZCB price
        zcb_prices[step] = zcbprice(n_years - dt*step, r_t, h)       
 
    # the rates generated (according to the periods_per_year) are transformed back to annual rates
    rates = pd.DataFrame( compounding_rate(rates) )
    zcb_prices = pd.DataFrame( zcb_prices )

    return rates, zcb_prices

def bond_cash_flows(principal=100, maturity=10, coupon_rate=0.03, coupons_per_year=2):
    '''
    Generates a pd.Series of cash flows of a regular bond. Note that:
    '''
    # total number of coupons 
    n_coupons = round(maturity * coupons_per_year)
    
    # coupon amount 
    coupon_amount = (coupon_rate / coupons_per_year) * principal 
    
    # Cash flows
    cash_flows = pd.DataFrame(coupon_amount, index = np.arange(1,n_coupons+1), columns=[0])
    cash_flows.iloc[-1] = cash_flows.iloc[-1] + principal 
        
    return cash_flows
    
def bond_price(principal=100, maturity=10, coupon_rate=0.02, coupons_per_year=2, ytm=0.03, cf=None):
    '''
    Return the price of regular coupon-bearing bonds
    Note that:
    - the maturity is intended as an annual variable (e.g., for a 6-months bond, maturity is 0.5);
    - the principal (face value) simply corresponds to the capital invested in the bond;
    - the coupon_rate has to be an annual rate;
    - the coupons_per_year is the number of coupons paid per year;
    - the ytm is the yield to maturity: then ytm divided by coupons_per_year gives the discount rate of cash flows
    The ytm variable can be both a single value and a pd.DataFrame. 
    In the former case, a single bond price is computed. In addition, if the flux of cash flows is computed beforehand, 
    the method can takes it as input and avoid recomputing it.
    In the latter case, the dataframe is intended as a t-by-scenarios matrix, where t are the dates and scenarios denotes
    the number of rates scenario in input. Here, for each scenario, single bond prices are computed according to different ytms.
    '''
    # single bond price 
    def single_price_bond(principal=principal, maturity=maturity, coupon_rate=coupon_rate, coupons_per_year=coupons_per_year, ytm=ytm, cf=cf):
        if cf is None:            
            # compute the bond cash flow on the fly
            cf = bond_cash_flows(maturity=maturity, principal=principal, coupon_rate=coupon_rate, coupons_per_year=coupons_per_year)             
        bond_price = present_value(cf, ytm/coupons_per_year)[0]
        return bond_price
    
    if isinstance(ytm,pd.Series):
        raise TypeError("Expected pd.DataFrame or a single value for ytm")

    if isinstance(ytm,pd.DataFrame):
        # ytm is a dataframe of rates for different scenarios 
        n_scenarios = ytm.shape[1]
        bond_price  = pd.DataFrame()
        # we have a for over each scenarios of rates (ytms)
        for i in range(n_scenarios):
            # for each scenario, a list comprehension computes bond prices according to ytms up to time maturity minus 1
            prices = [single_price_bond(principal=principal, maturity=maturity - t/coupons_per_year, coupon_rate=coupon_rate,
                                        coupons_per_year=coupons_per_year, ytm=y, cf=cf) for t, y in zip(ytm.index[:-1], ytm.iloc[:-1,i]) ] 
            bond_price = pd.concat([bond_price, pd.DataFrame(prices)], axis=1)
        # rename columns with scenarios
        bond_price.columns = ytm.columns
        # concatenate one last row with bond prices at maturity for each scenario
        bond_price = pd.concat([ bond_price, 
                                 pd.DataFrame( [[principal+principal*coupon_rate/coupons_per_year] * n_scenarios], index=[ytm.index[-1]]) ], 
                                axis=0)
        return bond_price 
    else:
        # base case: ytm is a value and a single bond price is computed 
        return single_price_bond(principal=principal, maturity=maturity, coupon_rate=coupon_rate, 
                                 coupons_per_year=coupons_per_year, ytm=ytm, cf=cf)        

def bond_returns(principal, bond_prices, coupon_rate, coupons_per_year, periods_per_year, maturity=None):
    '''
    Computes the total return of a coupon-paying bond. 
    The bond_prices can be a pd.DataFrame of bond prices for different ytms and scenarios 
    as well as a single bond price for a fixed ytm. 
    In the first case, remind to annualize the computed returns.
    In the latter case, the maturity of the bond has to passed since cash-flows needs to be recovered. 
    Moreover, the computed return does not have to be annualized.
    '''
    if isinstance(bond_prices, pd.DataFrame):
        coupons = pd.DataFrame(data=0, index=bond_prices.index, columns=bond_prices.columns)
        last_date = bond_prices.index.max()
        pay_date = np.linspace(periods_per_year/coupons_per_year, last_date, int(coupons_per_year*last_date/periods_per_year), dtype=int  )
        coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
        tot_return = (bond_prices + coupons)/bond_prices.shift(1) - 1 
        return tot_return.dropna()
    else:
        cf = bond_cash_flows(principal=principal, maturity=maturity, coupon_rate=coupon_rate, coupons_per_year=coupons_per_year) 
        tot_return = ( cf.sum() / bond_prices )**(1/maturity) - 1
        return tot_return[0]

def mac_duration(cash_flows, discount_rate):
    '''
    Computed the Macaulay duration of an asset involving regular cash flows a given discount rate
    Note that if the cash_flows dates are normalized, then the discount_rate is simply the YTM. 
    Otherwise, it has to be the YTM divided by the coupons per years.
    '''
    if not isinstance(cash_flows,pd.DataFrame):
        raise ValueError("Expected a pd.DataFrame of cash_flows")

    dates = cash_flows.index

    # present value of single cash flows (discounted cash flows)
    discount_cf = discount( dates, discount_rate ) * cash_flows
    
    # weights: the present value of the entire payment, i.e., discount_cf.sum() is equal to the principal 
    weights = discount_cf / discount_cf.sum()
    
    # sum of weights * dates
    return ( weights * pd.DataFrame(dates,index=weights.index) ).sum()[0]

# ---------------------------------------------------------------------------------
# Liability driven strategies 
# ---------------------------------------------------------------------------------
def ldi_mixer(psp_rets, lhp_rets, allocator, **kwargs):
    '''
    Liability-Driven Investing strategy allocator. 
    The method takes in input the returns of two portoflios, the returns of the PSP assets, psp_rets, 
    and the returns of the LHP assets, lhp_rets. 
    The allocator is the name of the allocator method to be used and which returns the weight in psp_rets, 
    while the extra arguments which are te arguments used by the allocator.
    The method returns a dataframe consisting of the weighted average 
    of the returns.
    '''
    if not psp_rets.shape == lhp_rets.shape:
        # make sure shapes coincides
        raise ValueError("Expected psp_rets and lhp_rets to be the same shape")
    
    weights = allocator(psp_rets, lhp_rets, **kwargs)
    
    if not weights.shape == psp_rets.shape:
        # make sure shapes coincides
        raise ValueError("Weight shapes do not match psp_rets (and lhp_rets) shape")
        
    return weights*psp_rets + (1-weights)*lhp_rets

def ldi_fixed_allocator(psp_rets, lhp_rets, w1, **kwargs):
    '''
    Fixed-mix strategy allocation between two asset returns.
    Here, psp_rets and lhp_rets are the returns of the PSP assets and the LHP assets, respectively, 
    and w1 is the weight in psp_rets.
    The method returns a dataframe consisting of the weight w1.
    The method is called by the LDI_MIXER method.
    '''
    return pd.DataFrame(data=w1, index=psp_rets.index, columns=psp_rets.columns)

def ldi_glidepath_allocator(psp_rets, lhp_rets, start=1, end=0):
    '''
    Glide path fixed-mix strategy allocation between two asset returns.
    Here, psp_rets and lhp_rets are the returns of the PSP assets and the LHP assets, respectively, 
    and w1 is the weight in psp_rets.
    The method returns a dataframe consisting of linearly spaced weights (in psp_rets) 
    from start to end, which have to be given in input. 
    The method is called by the LDI_MIXER method.
    '''
    # allocating linearly spaced weigths from strat to end 
    single_path = pd.DataFrame(data=np.linspace(start, end, num=psp_rets.shape[0]))
    paths = pd.concat( [single_path]*psp_rets.shape[1], axis=1 )
    paths.index   = psp_rets.index
    paths.columns = psp_rets.columns
    return paths

def ldi_floor_allocator(psp_rets, lhp_rets, zcb_price, floor, m=3):
    '''
    Allocate weights to PSP and LHP with the goal to provide exposure to the upside 
    of the PSP without going violating the floor. 
    The method uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple 
    of a cushion in the PSP. The cushion is computed by using a floor value to be 
    equal to the floor times the prices of the ZCB.
    The method return a pd.DataFrame containing the weights in the PSP 
    '''
    if not zcb_price.shape == psp_rets.shape:
        raise ValueError("PSP rets and ZCB prices must have the same shape")
        
    dates, n_scenarios = psp_rets.shape
    account_value  = np.repeat(1,n_scenarios)
    floor_value    = np.repeat(1,n_scenarios)
    weight_history = pd.DataFrame(index=psp_rets.index, columns=psp_rets.columns)
    for date in range(dates):
        floor_value = floor * zcb_price.iloc[date]
        cushion = (account_value - floor_value) / account_value
        # weights in the PSP and LHP 
        psp_w = (m * cushion).clip(0,1)
        lhp_w = 1 - psp_w
        # update
        account_value = psp_w*account_value*(1 + psp_rets.iloc[date]) + lhp_w*account_value*(1 + lhp_rets.iloc[date])
        weight_history.iloc[date] = psp_w
    return weight_history

def ldi_drawdown_allocator(psp_rets, lhp_rets, maxdd=0.2):
    '''
    Allocate weights to PSP and LHP with the goal to provide exposure to the upside 
    of the PSP without going violating the floor. 
    The method uses a CPPI-style dynamic risk budgeting algorithm with a drawdown constraint: 
    we investing a multiple m (equal the inverse of the maxdd) of the cushion in the PSP. 
    The cushion is computed by using a floor value equal to (1-maxdd) times the current peak.
    The method return a pd.DataFrame containing the weights in the PSP.
    Also look at the LDI_FLOOR_ALLOCATOR.
    '''
    if not psp_rets.shape == lhp_rets.shape:
        raise ValueError("PSP and LHP returns must have the same shape")
        
    # define the multipler as the inverse of the maximum drawdown
    m = 1 / maxdd
    dates, n_scenarios = psp_rets.shape
    account_value  = np.repeat(1,n_scenarios)
    floor_value    = np.repeat(1,n_scenarios)
    peak_value     = np.repeat(1,n_scenarios)
    weight_history = pd.DataFrame(index=psp_rets.index, columns=psp_rets.columns)
    
    for date in range(dates):
        floor_value = (1 - maxdd)*peak_value
        cushion = (account_value - floor_value) / account_value
        # weights in the PSP and LHP 
        psp_w = (m * cushion).clip(0,1)
        lhp_w = 1 - psp_w
        # update
        account_value = psp_w*account_value*(1 + psp_rets.iloc[date]) + lhp_w*account_value*(1 + lhp_rets.iloc[date])
        peak_value = np.maximum(peak_value, account_value)
        weight_history.iloc[date] = psp_w
    return weight_history 

# ---------------------------------------------------------------------------------
# Factor and Style analysis 
# ---------------------------------------------------------------------------------
def linear_regression(dep_var, exp_vars, alpha=True):
    '''
    Runs a linear regression to decompose the dependent variable into the explanatory variables 
    using statsmodels OLS method. 
    It returns the object of type statsmodel's RegressionResults on which we can call on it:
    - .summary() to print a full summary
    - .params for the coefficients
    - .tvalues and .pvalues for the significance levels
    - .rsquared_adj and .rsquared for quality of fit
    Note that exp.vars can be both a pd.DataFrame a np.array.
    '''
    if alpha:
        # the OLS methods assume a bias equal to 0, hence a specific variable for the bias has to be given
        if isinstance(exp_vars,pd.DataFrame):
            exp_vars = exp_vars.copy()
            exp_vars["Alpha"] = 1
        else:
            exp_vars = np.concatenate( (exp_vars, np.ones((exp_vars.shape[0],1))), axis=1 )
    return sm.OLS(dep_var, exp_vars).fit()

def capm_betas(ri, rm):
    '''
    Returns the CAPM factor exposures beta for each asset in the ri pd.DataFrame, 
    where rm is the pd.DataFrame (or pd.Series) of the market return (not excess return).
    The betas are defined as:
      beta_i = Cov(r_i, rm) / Var(rm)
    with r_i being the ith column (i.e., asset) of DataFrame ri.
    '''
    market_var = ( rm.std()**2 )[0]
    betas = []
    for name in ri.columns:
        cov_im = pd.concat( [ri[name],rm], axis=1).cov().iloc[0,1]
        betas.append( cov_im / market_var )
    return pd.Series(betas, index=ri.columns)

def tracking_error(r_a, r_b):
    '''
    Returns the tracking error between two return series. 
    This method is used in Sharpe Analysis minimization problem.
    See STYLE_ANALYSIS method.
    '''
    return ( ((r_a - r_b)**2).sum() )**(0.5)

def style_analysis_tracking_error(weights, ref_r, bb_r):
    '''
    Sharpe style analysis objective function.
    Returns the tracking error between the reference returns
    and a portfolio of building block returns held with given weights. 
    '''
    return tracking_error(ref_r, (weights*bb_r).sum(axis=1))

def style_analysis(dep_var, exp_vars):
    '''
    Sharpe style analysis optimization problem.
    Returns the optimal weights that minimizes the tracking error between a portfolio 
    of the explanatory (return) variables and the dependent (return) variable.
    '''
    # dep_var is expected to be a pd.Series
    if isinstance(dep_var,pd.DataFrame):
        dep_var = dep_var[dep_var.columns[0]]
    
    n = exp_vars.shape[1]
    init_guess = np.repeat(1/n, n)
    weights_const = {
        'type': 'eq',
        'fun': lambda weights: 1 - np.sum(weights)
    }
    solution = minimize(style_analysis_tracking_error, 
                        init_guess,
                        method='SLSQP',
                        options={'disp': False},
                        args=(dep_var, exp_vars),
                        constraints=(weights_const,),
                        bounds=((0.0, 1.0),)*n)
    weights = pd.Series(solution.x, index=exp_vars.columns)
    return weights

# ---------------------------------------------------------------------------------
# Covariance matrix estimators
# ---------------------------------------------------------------------------------
def sample_cov(r, **kwargs):
    '''
    Returns the sample covariance of the supplied series of returns (a pd.DataFrame) 
    '''
    if not isinstance(r,pd.DataFrame):
        raise ValueError("Expected r to be a pd.DataFrame of returns series")
    return r.cov()

def cc_cov(r, **kwargs):
    '''
    Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
    '''
    # correlation coefficents  
    rhos = r.corr()
    n = rhos.shape[0]
    # compute the mean correlation: since the matrix rhos is a symmetric with diagonals all 1, 
    # the mean correlation can be computed by:
    mean_rho = (rhos.values.sum() - n) / (n**2-n) 
    # create the constant correlation matrix containing 1 on the diagonal and the mean correlation outside
    ccor = np.full_like(rhos, mean_rho)
    np.fill_diagonal(ccor, 1.)
    # create the new covariance matrix by multiplying mean_rho*std_i*std_i 
    # the product of the stds is done via np.outer
    ccov = ccor * np.outer(r.std(), r.std())
    return pd.DataFrame(ccov, index=r.columns, columns=r.columns)

def shrinkage_cov(r, delta=0.5, **kwargs):
    '''
    Statistical shrinkage: it returns a covariance matrix estimator that shrinks between 
    the constant correlation and standard sample covariance estimators 
    '''
    samp_cov  = sample_cov(r, **kwargs)
    const_cov = cc_cov(r, **kwargs)
    return delta*const_cov + (1-delta)*samp_cov

# ---------------------------------------------------------------------------------
# Back-test weigthing schemes
# ---------------------------------------------------------------------------------
def weight_ew(r, cap_ws=None, max_cw_mult=None, microcap_thr=None, **kwargs):
    """
    Returns the weights of the Equally-Weighted (EW) portfolio based on the asset returns "r" as a DataFrame. 
    If the set of cap_ws is given, the modified scheme is computed, i.e., 
    microcaps are removed and a capweight tether applied.
    """
    ew = pd.Series(1/len(r.columns), index=r.columns)
    if cap_ws is not None:
        cw = cap_ws.loc[r.index[0]] # starting cap weight
        if microcap_thr is not None and microcap_thr > 0.0:
            # exclude microcaps according to the threshold    
            ew[ cw < microcap_thr ] = 0
            ew = ew / ew.sum()
        if max_cw_mult is not None and max_cw_mult > 0:
            # limit weight up to a multiple of capweight
            ew = np.minimum(ew, cw*max_cw_mult)
            ew = ew / ew.sum()
    return ew

def weight_cw(r, cap_ws, **kwargs):
    '''
    Returns the weights of the Cap-Weigthed (CW) portfolio based on the time series of capweights
    '''
    return cap_ws.loc[r.index[0]]
    # which is equal to:
    # w = cap_ws.loc[r.index[0]]
    # return w / w.sum()
    # since cap_ws are already normalized
    
def weight_rp(r, cov_estimator=sample_cov, **kwargs):
    '''
    Produces the weights of the risk parity portfolio given a covariance matrix of the returns.
    The default coavariance estimator is the sample covariance matrix.
    '''
    est_cov = cov_estimator(r, **kwargs)
    return risk_parity_weigths(est_cov)  

def backtest_weight_scheme(r, window=36, weight_scheme=weight_ew, **kwargs):
    '''
    Backtests a given weighting scheme. Here:
    - r: asset returns to use to build the portfolio
    - window: the rolling window used
    - weight_scheme: the weighting scheme to use, it must the name of a 
    method that takes "r", and a variable number of keyword-value arguments
    '''
    n_periods = r.shape[0]
    windows = [ (start, start+window) for start in range(0,n_periods-window) ]
    weights = [ weight_scheme( r.iloc[win[0]:win[1]], **kwargs) for win in windows ]
    weights = pd.DataFrame(weights, index=r.iloc[window:].index, columns=r.columns)    
    returns = (weights * r).sum(axis=1,  min_count=1) #mincount is to generate NAs if all inputs are NAs
    return returns

def annualize_vol_ewa(r, decay=0.95, periods_per_year=12):
    '''
    Computes the annualized exponentially weighted average volatility of a 
    series of returns given a decay (smoothing) factor in input. 
    '''
    N = r.shape[0]
    times = np.arange(0,N,1)
    # compute the square error returns
    sq_errs = pd.DataFrame( ( r - r.mean() )**2 )
    # exponential weights
    weights = [ decay**(N-t) for t in times ] / np.sum(decay**(N-times))
    weights = pd.DataFrame(weights, index=r.index)
    # EWA
    vol_ewa = (weights * sq_errs).sum()**(0.5)
    # Annualize the computed volatility
    ann_vol_ewa = vol_ewa[0] * np.sqrt(periods_per_year)
    return ann_vol_ewa

def weight_minvar(r, cov_estimator=sample_cov, periods_per_year=12, **kwargs):
    '''
    Produces the weights of the Minimum Volatility Portfolio given a covariance matrix of the returns 
    '''
    est_cov = cov_estimator(r, **kwargs)
    ann_ret = annualize_rets(r, periods_per_year=12)
    return minimize_volatility(ann_ret, est_cov)

def weight_maxsharpe(r, cov_estimator=sample_cov, periods_per_year=12, risk_free_rate=0.03, **kwargs):
    '''
    Produces the weights of the Maximum Sharpe Ratio Portfolio given a covariance matrix of the returns 
    '''
    est_cov = cov_estimator(r, **kwargs)
    ann_ret = annualize_rets(r, periods_per_year=12)
    return maximize_shape_ratio(ann_ret, est_cov, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    
# ---------------------------------------------------------------------------------
# Black-Litterman model
# ---------------------------------------------------------------------------------
def implied_returns(covmat, weigths, delta=2.5):
    '''
    Computes the implied expected returns \Pi by reverse engineering the weights according to 
    the Black-Litterman model:
       \Pi = \delta \Sigma weigths
    Here, the inputs are:
    - delta, the risk aversion coefficient
    - covmat: variance-covariance matrix (N x N) as pd.DataFrame (\Sigma)
    - weigths: portfolio weights (N x 1) as pd.Series 
    The output is the \Pi returns pd.Series (N x 1) 
    '''
    imp_rets = delta * covmat.dot(weigths).squeeze() # to get a series from a 1-column dataframe
    imp_rets.name = 'Implied Returns'
    return imp_rets

def omega_uncertain_prior(covmat, tau, P):
    '''
    Returns the He-Litterman simplified Omega matrix in case the investor does not explicitly 
    quantify the uncertainty on the views. This matrix is going to be:
       \Omega := diag( P(\tau\Sigma)P^T ) 
    Inputs:
    - covmat: N x N covariance Matrix as pd.DataFrame (\Sigma)
    - tau: a scalar denoting the uncertainty of the CAPM prior
    - P: the Projection K x N matrix as a pd.DataFrame.
    The output is a P x P matrix as a pd.DataFrame, representing the Prior Uncertainties.
    '''
    he_lit_omega = P.dot(tau * covmat).dot(P.T)
    # Make a diag matrix from the diag elements of Omega
    return pd.DataFrame( np.diag(np.diag(he_lit_omega.values)), index=P.index, columns=P.index )

def black_litterman(w_prior, Sigma_prior, P, Q, Omega=None, delta=2.5, tau=0.02):
    '''
    Black-Litterman model.
    Computes the posterior expected returns and covariaces based on the original Black-Litterman model 
    using the Master formulas, where:
    - w_prior is the N x 1 pd.Series of prior weights
    - Sigma_prior is the N x N covariance matrix as a pd.DataFrame
    - P is the projection K x N matrix of weights portfolio views, a pd.DataFrame
    - Q is the K x 1 pd.Series of views
    - Omega is the K x K matrix as a pd.DataFrame (or None) representing the uncertainty of the views. 
      In particualar, if Omega=None, we assume that it is proportional to variance of the prior (see OMEGA_UNCERTAIN_PRIOR).
    - delta is the risk aversion (scalar)
    - tau represents the uncertainty of the CAPM prior (scalar)
    '''
    if Omega is None:
        Omega = omega_uncertain_prior(Sigma_prior, tau, P)
    # Force w.prior and Q to be column vectors
    #w_prior = as_colvec(w_prior)
    #Q = as_colvec(Q)
    
    # number of assets
    N = w_prior.shape[0]

    # number of views
    K = Q.shape[0]
    
    # First, reverse-engineer the weights to get \Pi = \delta\Sigma\w^{prior}
    Pi = implied_returns(Sigma_prior,  w_prior, delta)
        
    # Black-Litterman posterior estimate (Master Formulas), using the versions that do not require Omega to be inverted
    invmat   = inv( P.dot(tau * Sigma_prior).dot(P.T) + Omega )
    mu_bl    = Pi + (tau * Sigma_prior).dot(P.T).dot(invmat.dot(Q - P.dot(Pi).values))
    sigma_bl = Sigma_prior + (tau * Sigma_prior) - (tau * Sigma_prior).dot(P.T).dot(invmat).dot(P).dot(tau * Sigma_prior)
    return (mu_bl, sigma_bl)

# ---------------------------------------------------------------------------------
# Risk contributions analysis 
# ---------------------------------------------------------------------------------
def enc(weigths):
    '''
    Computes the Effective Number of Constituents (ENC) given an input 
    vector of weights of a portfolio.
    '''
    return (weigths**2).sum()**(-1)

def encb(risk_contrib):
    '''
    Computes the Effective Number of Correlated Bets (ENBC) given an input 
    vector of portfolio risk contributions.
    '''
    return (risk_contrib**2).sum()**(-1)

def portfolio_risk_contributions(weigths, matcov):
    '''
    Compute the contributions to risk of the asset constituents of a portfolio, 
    given a set of portfolio weights and a covariance matrix.
    The input weigths has to be either a np.array or a pd.Series
    while matcov must be a pd.DataFrame.
    '''
    portfolio_var = portfolio_volatility(weigths, matcov)**2
    # marginal contribution of each constituent: the "@" operator is equal to np.dot but return a pd.Series
    marginal_contrib = matcov @ weigths
    # vector of risk contributions
    risk_contrib = np.multiply(marginal_contrib, weigths.T) / portfolio_var
    return risk_contrib

def msd_risk_contrib(weights, target_risk, mat_cov):
    '''
    Returns the Mean Squared Difference between the given target risk contribution vector and the 
    risk contributions due to current weigths. 
    This method implements the objective function to minimize which is called by the quadratic 
    minimizer PORTFOLIO_RISK_CONTRIB_OPTIMIZER.
    '''
    w_risk_contribs = portfolio_risk_contributions(weights, mat_cov)
    msd = (w_risk_contribs - target_risk)**2 
    return msd.sum() #mean()

def portfolio_risk_contrib_optimizer(target_risk_contrib, mat_cov):
    '''
    Returns the weights of the portfolio whose asset risk contributions are 
    as close as possible to the input target_risk contribution vector. 
    The input target_risk must be either a pd.Series or np.array while 
    mat_cov must be a pd.DataFrame.
    '''
    n = mat_cov.shape[0]
    init_guess = np.repeat(1/n, n)

    # constraint on weights
    weights_sum_to_one = {
        'type': 'eq',
        'fun': lambda weights: 1 - np.sum(weights)
    }
        
    weights = minimize(msd_risk_contrib, 
                       init_guess,
                       args=(target_risk_contrib, mat_cov), 
                       method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_one,),
                       bounds=((0.0, 1.0),)*n )
    return weights.x

def risk_parity_weigths(mat_cov):
    '''
    Returns the weights of the portfolio that equalizes the asset risk contributions 
    of the constituents, that is, return the risk parity portfolio. 
    The method uses the quadratic optimizer PORTFOLIO_RISK_CONTRIB_OPTIMIZER with a target risk 
    vector of 1/N, where N is the number of asset since we know that this is the value for 
    which the ENCB=N.
    '''
    n = mat_cov.shape[0]
    weigths = portfolio_risk_contrib_optimizer(target_risk_contrib=np.repeat(1/n,n), mat_cov=mat_cov)
    return pd.Series(weigths, index=mat_cov.index)

# ---------------------------------------------------------------------------------
# Auxiliary methods 
# ---------------------------------------------------------------------------------
def as_colvec(x):
    '''
    In order to consistently use column vectors, this method takes either a np.array or a np.column 
    matrix (i.e., a column vector) and returns the input data as a column vector.
    '''
    if (x.ndim == 2):
        return x
    else:
        return np.expand_dims(x, axis=1)
    
def inverse_df(d):
    '''
    Inverse of a pd.DataFrame (i.e., inverse of dataframe.values)
    '''
    return pd.DataFrame( inv(d.values), index=d.columns, columns=d.index)

def insert_first_row_df(df, row):
    '''
    The method inserts a row at the beginning of a given dataframe and shift by one existing rows.
    The input row has to be either a single element (in case of 1-column dataframe) or 
    a dictionary in case of multi-column dataframe.
    
    Example:
        df  = pd.DataFrame([1, 2, 3])
        row = 0.5
        df  = insert_first_row_df(df, row)
    Example:    
        df  = pd.DataFrame([[1,2],[34,12]], columns=["A","B"])
        row = {"A":-34, "B":443}
        df  = insert_first_row_df(df, row)        
    '''
    df.loc[-1] = row
    df.index = df.index + 1
    return df.sort_index()