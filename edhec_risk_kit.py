import pandas as pd
import numpy as np
import scipy.stats
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

    
def drawdown(returns: pd.Series):
    '''
    Drawdown: takes in input the returns of an asset and returns a dataframe containing 
    1. the associated wealth index (for an hypothetical investment of $1000) 
    2. all previous peaks 
    3. the drawdowns
    '''
    wealth_index   = 1000 * (1 + returns).cumprod()
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



def effront_two_assets(n_portfolios, rets, covmat, periods_per_year, risk_free_rate=0.0):
    '''
    Return the efficient frontiers for a portfolio of two assets. 
    It returns a dataframe containing the volatilitis, the returns, the sharpe ratios of 
    the portfolios as well as a plot of the efficient frontier.
    The variable periods_per_year can be, e.g., 12, 52, 252, in case of yearly, weekly, and daily data.
    '''
    weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_portfolios)]

    # portfolio returns
    ann_rets      = annualize_rets(rets, periods_per_year)
    portfolio_ret = [portfolio_return(w, ann_rets) for w in weights]
    
    # portfolio volatility
    vols          = [portfolio_volatility(w, covmat) for w in weights] 
    portfolio_vol = [annualize_vol(v, periods_per_year) for v in vols]
    
    # portfolio sharpe ratio
    portfolio_spr = [sharpe_ratio(r, risk_free_rate, periods_per_year, v=v) for r,v in zip(portfolio_ret,portfolio_vol)]
    
    df = pd.DataFrame({"volatility": portfolio_vol,
                       "return": portfolio_ret,
                       "sharpe ratio": portfolio_spr})
    return df, df.plot.line(x="volatility", y="return", style=".-", grid=True)











def efficient_frontier(n_portfolios, rets, covmat, periods_per_year, risk_free_rate=0.0, iplot=False):
    '''
    Returns (and plots) the efficient frontiers for a portfolio of rets.shape[1] assets. 
    The method returns a dataframe containing the volatilities, returns, and sharpe ratios of 
    the portfolios as well as a plot of the efficient frontier in case iplot=True.
    The variable periods_per_year can be, e.g., 12, 52, 252, in case of yearly, weekly, and daily data.
    '''    
    ann_rets = annualize_rets(rets, periods_per_year)
    
    # generates optimal weights of porfolios lying of the efficient frontiers
    weights = optimal_weights(n_portfolios, ann_rets, covmat, periods_per_year) 

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
    if iplot:
        return df, df.plot.line(x="volatility", y="return", style=".-", grid=True, label="Efficient frontier")
    else: 
        return df

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

def optimal_weights(n_points, rets, covmatrix, periods_per_year):
    '''
    Returns a set of n_points optimal weights corresponding to portfolios (of the efficient frontier) 
    with minimum volatility constructed by fixing n_points target returns. 
    The weights are obtaine by solving the minimization problem for the volatility. 
    '''
    target_rets = np.linspace(rets.min(), rets.max(), n_points)    
    weights = [minimize_volatility(rets, covmatrix, target) for target in target_rets]
    return weights