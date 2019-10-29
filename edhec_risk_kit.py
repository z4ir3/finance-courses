import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def get_ffme_returns():
    '''
    Returns the French-Fama dataset for the returns of the bottom and top 
    deciles (Low 10 (Small Caps) and Hi 10 (Large Caps)) of US stocks
    '''
    rets = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv", index_col=0, parse_dates=True, na_values=-99.99)
    # Divide by 100, since they are returns, and change the index to datatime
    rets = rets[["Lo 10", "Hi 10"]] / 100
    rets.index = pd.to_datetime(rets.index, format="%Y%m") #.to_period("M") forces the index to be monthly period...
    return rets 
   
def get_hfi_returns():
    '''
    Returns the EDHEC Hedge Funds Index returns
    '''
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv", index_col=0, parse_dates=True, na_values=-99.99)
    # Divide by 100, since they are returns, and change the index to datatime
    hfi = hfi / 100
    # the index is already of type datetime
    return hfi 

def get_ind_returns():
    '''
    Load and format the Ken French 30 Industry portfolios value weighted monthly returns 
    '''
    ind = pd.read_csv("data/ind30_m_vw_rets.csv", index_col=0, parse_dates=True)
    # Divide by 100, since they are returns
    ind = ind / 100
    # the index is not yet of type datetime
    ind.index = pd.to_datetime(ind.index, format="%Y%m") #.to_period("M") forces the index to be monthly period...
    # there will be blank spaces in the columns names
    ind.columns = ind.columns.str.strip()
    return ind


def get_ind_nfirms():
    '''
    Load and format the Ken French 30 Industry number of firms dataset
    '''
    ind = pd.read_csv("data/ind30_m_nfirms.csv", index_col=0, parse_dates=True)
    # the index is not yet of type datetime
    ind.index = pd.to_datetime(ind.index, format="%Y%m") #.to_period("M") forces the index to be monthly period...
    # there will be blank spaces in the columns names
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_size():
    '''
    Load and format the Ken French 30 Industry average (market cap) size
    '''
    ind = pd.read_csv("data/ind30_m_size.csv", index_col=0, parse_dates=True)
    # the index is not yet of type datetime
    ind.index = pd.to_datetime(ind.index, format="%Y%m") #.to_period("M") forces the index to be monthly period...
    # there will be blank spaces in the columns names
    ind.columns = ind.columns.str.strip()
    return ind


def get_total_market_index_returns():
    '''
    Computes the (total market) returns from the Ken French 30 Industry portfolio
    '''
    # Load the portfolio returns, number of firms, and average size
    ind_rets   = get_ind_returns()
    ind_nfirms = get_ind_nfirms()
    ind_size   = get_ind_size()     
    # compute the market capitalization of each industry sector
    ind_mkt_cap = ind_nfirms * ind_size
    # compute the total market capitalization
    total_mkt_cap = ind_mkt_cap.sum(axis=1)
    # compute the single market capitalizations as a percentage of the total market cap
    ind_cap_weights = ind_mkt_cap.divide(total_mkt_cap, axis=0)
    # finally, computes the total market returns         
    return (ind_cap_weights * ind_rets).sum(axis=1)


def compound_returns(s, start=100):
    '''
    Compound a Dataframe or Series of returns from an initial default value equal to 100.
    In the former case, the method compounds the returns for every column (Series) by using pd.aggregate. 
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
    return ( ((s - s.mean()) / s.std())**3 ).mean()

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
    return s[s<0].std() 

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
        za = za + (za**2 - 1)*S/6 + (za**3 - 3*za)*(K-3)/24 - (2*za**3 - 5*za)*(S**2)/34    
    return -( s.mean() + za * s.std() )

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
    case of yearly, weekly, and daily data.
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
    case of yearly, weekly, and daily data.
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
    The variable periods_per_year can be, e.g., 12, 52, 252, in case of yearly, weekly, and daily data.
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
                ax.scatter([vol], [ret], marker="o", color="g", label="Highest sharpe ratio portfolio")
                ax.legend()
        if mvp:
            # Plot the equally weighted portfolio:
            w   = minimize_volatility(ann_rets, covmat)
            ret = portfolio_return(w, ann_rets)
            vol = annualize_vol( portfolio_volatility(w,covmat), periods_per_year)
            spr = sharpe_ratio(ret, risk_free_rate, periods_per_year, v=vol)
            df  = append_row_df(df,vol,ret,spr,w)
            ax.scatter([vol], [ret], color="midnightblue", marker="o", label="Minimum volatility portfolio")
            ax.legend()  
        if ewp:
            # Plot the equally weighted portfolio:
            w   = np.repeat(1/ann_rets.shape[0], ann_rets.shape[0])
            ret = portfolio_return(w, ann_rets)
            vol = annualize_vol( portfolio_volatility(w,covmat), periods_per_year)
            spr = sharpe_ratio(ret, risk_free_rate, periods_per_year, v=vol)
            df  = append_row_df(df,vol,ret,spr,w)
            ax.scatter([vol], [ret], color="goldenrod", marker="o", label="Equally weighted portfolio")
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
        stats     = {
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
     
def optimal_weights(n_points, rets, covmatrix, periods_per_year):
    '''
    Returns a set of n_points optimal weights corresponding to portfolios (of the efficient frontier) 
    with minimum volatility constructed by fixing n_points target returns. 
    The weights are obtaine by solving the minimization problem for the volatility. 
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




def discount(t, r):
    '''
    Compute the price of a pure discount bond that pays 1 at time t (year),
    given an interest rate (return) r. That is, considering FV = 1 at time t, 
    want to obtain the PV given r, i.e., PV = FV/(1+r)^t = 1/(1+r)^t.
    ''' 
    return 1 / (1 + r)**t

    
    
    
    
    
    
    
    
    
    
    
def nominal2annual_rate(r, periods_per_year):
    return (1 + r/periods_per_year)**periods_per_year - 1

def nominal2annual_rate_gen(r):
    return np.exp(r) - 1

def annual2nomimal_rate(R, periods_per_year):
    return periods_per_year * ( (1 + R)**(1/periods_per_year) - 1 )

def annual2nomimal_rate_gen(R):
    return np.log( 1 + R )     
    
    
    
    
    
    
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
    r0 = annual2nomimal_rate_gen(r0)
    
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
 
    rates = pd.DataFrame( nominal2annual_rate_gen(rates) )
    zcb_prices = pd.DataFrame( zcb_prices )

    return rates, zcb_prices

    
    
    
    
    
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