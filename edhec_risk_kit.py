import pandas as pd
import numpy as np
import scipy.stats

def get_ffme_returns():
    '''
    Returns the French-Fama dataset for the returns of the bottom and top deciles (Low 10 and Hi 10)
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
    #hfi.index = hfi.index.to_period("M") #.to_period("M") forces the index to be monthly period...
    return hfi 
    
def drawdown(returns: pd.Series):
    '''
    Computing the Drawdown: takes in input a series of asset returns and returns a dataframe containing 
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
    There is also the function scipy.stats.skew()
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
    Simply return the Excess Kurtosis, i.e., Kurtosis minus 3
    '''
    return kurtosis(s) - 3

def is_normal(s, level=0.01):
    '''
    Jarque-Bera test to see if a series (of returns) is normally distributed
    It returns True or False according to wheter the p value is larger than the level
    '''
    statistic, pvalue = scipy.stats.jarque_bera( s )
    return pvalue > level

def semivolatility(s):
    '''
    Returns the semivolatility of a series, i.e., the volatility of
    negative returns
    '''
    return s[s<0].std() 

def var_historic(s, alpha=0.05):
    '''
    Returns the (1-alpha)% VaR using historical method. 
    By default it computes the 95% VaR.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the VaR for every column (Series) by using pd.aggregate
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( var_historic, alpha=alpha )
    elif isinstance(s, pd.Series):
        return - np.percentile(s, alpha*100)
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
        
def var_gaussian(s, alpha=0.05, cf=False):
    '''
    Returns the (1-alpha)% VaR using the parametric Gaussian method. 
    By default it computes the 95% VaR.
    The variable "cf" stands for Cornish-Fisher. If True, the method computes the 
    modified VaR using the Cornish-Fisher expansion of quantiles.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the VaR for every column (Series).
    '''
    # alpha-quantile of Gaussian distribution 
    za = scipy.stats.norm.ppf(alpha,0,1) 
    if cf:
        S = skewness(s)
        K = kurtosis(s)
        za = za + (za**2 - 1)*S/6 + (za**3 - 3*za)*(K-3)/24 - (2*za**3 - 5*za)*(S**2)/34    
    return -( s.mean() + za * s.std() )

def cvar_historic(s, alpha=0.05):
    '''
    Computes the Conditional Var (based on historical method)
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( cvar_historic, alpha=alpha )
    elif isinstance(s, pd.Series):
        # find the returns which are less than (the historic) VaR
        mask = s < -var_historic(s, alpha=alpha)
        # and of them, take the mean 
        return -s[mask].mean()
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
