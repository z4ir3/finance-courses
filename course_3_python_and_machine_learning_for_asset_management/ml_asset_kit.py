import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime as dt
import cvxpy as cp
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
from scipy.optimize import minimize
from pandas_datareader import data 

sys.path.append("../")
import edhec_risk_kit as erk


def path_to_data_folder():
    return "/Users/mariacristinasampaolo/Documents/python/git-tracked/finance-courses/course_3_python_and_machine_learning_for_asset_management/data/" 

def get_factors_and_assets():
    '''
    Returns a set of factors (from 1985-01 to 2018-09)
    '''
    filepath = path_to_data_folder() + "Data_Oct2018_v2.csv"
    factors = pd.read_csv(filepath, index_col=0, parse_dates=True)
    factors.index = pd.to_datetime(factors.index, format="%Y%m").to_period("M") #.to_period("M") forces the index to be monthly period...
    return factors

# ------------------------------------------------------------------------------------------------------
# Linear regression
# ------------------------------------------------------------------------------------------------------
def linear_regression_sk(dep_var, exp_vars):
    '''
    Runs a linear regression to decompose the dependent variable into the explanatory variables 
    using scikit-learn LinearRegression() method
    It returns the object lm:
    - lm.coef_ to print the betas
    - lm.intercept to print the intercept alpha
    Note that exp.vars can be both a pd.DataFrame a np.array.
    '''            
    lm = LinearRegression(fit_intercept=True)
    return lm.fit(exp_vars, dep_var)

# ------------------------------------------------------------------------------------------------------
# Lasso regression
# ------------------------------------------------------------------------------------------------------
def lasso_regression_sk(dep_var, exp_vars, lambdapar=1):
    '''
    Runs a linear regression to decompose the dependent variable into the explanatory variables 
    using scikit-learn Lasso() method.
    It returns the object lm:
    - lm.coef_ to print the betas
    - lm.intercept to print the intercept alpha
    Note that exp.vars can be both a pd.DataFrame a np.array.
    '''            
    lm = Lasso(alpha=lambdapar/(2*dep_var.shape[0]), fit_intercept=True)
    return lm.fit(exp_vars, dep_var)

def cross_val_lasso_regression(dep_var, exp_vars, lambda_max=0.25, n_lambdas=100, n_folds=10, rs=None):
    '''
    Gridsearch Cross-Validation of Lasso regression.
    Recall that the best lambda is given by: best_lambda = best_alpha*2*exp_vars.shape[0], 
    where best_alpha = gsCV.best_params_["alpha"]
    '''
    # setup the estimator 
    if rs is None: 
        lasso_test = Lasso(fit_intercept=True)
    else:
        lasso_test = Lasso(random_state=rs, fit_intercept=True)
    
    # setup parameters
    alpha_max = lambda_max / (2*exp_vars.shape[0])
    alphas = np.linspace(1e-6, alpha_max, n_lambdas)
    alphas = {'alpha': alphas}

    # Grid Search Cross Validation
    gsCV = GridSearchCV(estimator=lasso_test, param_grid=alphas, cv=n_folds, refit=True)
    gsCV.fit(exp_vars, dep_var)
    
    return gsCV

# ------------------------------------------------------------------------------------------------------
# Ridge regression
# ------------------------------------------------------------------------------------------------------
def ridge_regression_sk(dep_var, exp_vars, lambdapar=1):
    '''
    Runs a linear regression to decompose the dependent variable into the explanatory variables 
    using scikit-learn Ridge() method.
    It returns the object lm:
    - lm.coef_ to print the betas
    - lm.intercept to print the intercept alpha
    Note that exp.vars can be both a pd.DataFrame a np.array.
    '''            
    lm = Ridge(alpha=lambdapar, fit_intercept=True)
    return lm.fit(exp_vars, dep_var)

def cross_val_ridge_regression(dep_var, exp_vars, lambda_max=0.25, n_lambdas=100, n_folds=10, rs=None):
    '''
    Gridsearch Cross-Validation of Ridge regression.
    Recall that the best lambda is given by: best_lambda = best_alpha*2*exp_vars.shape[0], 
    where best_alpha = gsCV.best_params_["alpha"]
    '''
    # setup the estimator 
    if rs is None: 
        ridge_test = Ridge(fit_intercept=True)
    else:
        ridge_test = Ridge(random_state=rs, fit_intercept=True)
    
    # setup parameters
    alpha_max = lambda_max
    alphas = np.linspace(1e-6, alpha_max, n_lambdas)
    alphas = {'alpha': alphas}

    # Grid Search Cross Validation
    gsCV = GridSearchCV(estimator=ridge_test, param_grid=alphas, cv=n_folds, refit=True)
    gsCV.fit(exp_vars, dep_var)
    
    return gsCV

# ------------------------------------------------------------------------------------------------------
# Elastic Net regression
# ------------------------------------------------------------------------------------------------------
def elasticnet_regression_sk(dep_var, exp_vars, lambdapar=0.25, l1_ratio=0.5):
    '''
    Runs a linear regression to decompose the dependent variable into the explanatory variables 
    using scikit-learn ElasticNet() method.
    It returns the object lm:
    - lm.coef_ to print the betas
    - lm.intercept to print the intercept alpha
    Note that exp.vars can be both a pd.DataFrame a np.array.
    '''            
    alpha = lambdapar / (2*exp_vars.shape[0])
    lm = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True)
    return lm.fit(exp_vars, dep_var)

def cross_val_elasticnet_regression(dep_var, exp_vars, lambda_max=0.25, n_lambdas=50, l1_ratio_max=0.99, n_l1ratio=50, n_folds=10, rs=None):
    '''
    Gridsearch Cross-Validation of Elastic Net regression.
    Recall that the best lambda_1 and best lambda_2 are given by:
    - best lambda_1 = best_alpha*2*exp_vars.shape[0]*best_L1ratio      is given by: best_lambda = best_alpha*2*exp_vars.shape[0], 
    - best lambda_2 = best_alpha*exp_vars.shape[0]*(1-best_L1ratio)  
    where:
      best_alpha = gsCV.best_params_["alpha"]
      best_L1ratio = gsCV.best_params_["l1_ratio"]
    '''
    # setup the estimator 
    if rs is None: 
        elastic_net_test = ElasticNet(fit_intercept=True)
    else:
        elastic_net_test = ElasticNet(random_state=rs, fit_intercept=True)
    
    # setup parameters
    alpha_max = lambda_max / (2*exp_vars.shape[0])
    alphas    = np.linspace(1e-6, alpha_max, n_lambdas)
    l1_ratios = np.linspace(1e-6, l1_ratio_max, n_l1ratio)
    params    = {'alpha': alphas, 'l1_ratio': l1_ratios}

    # Grid Search Cross Validation
    gsCV = GridSearchCV(estimator=elastic_net_test, param_grid=params, cv=n_folds, refit=True)
    gsCV.fit(exp_vars, dep_var)
    
    return gsCV 

def recover_regression_bestpar_from_gsCV(gsCV, data, reg_type):
    '''
    Return the best parameter found by the gridsearch cross-validation process 
    for the given input gsCV model that can be of reg_type:
    - lasso
    - ridge
    - elasticnet
    '''
    if reg_type == "lasso":
        best_alpha  = gsCV.best_params_["alpha"]    
        best_lambda = best_alpha * 2 * data.shape[0]
        print("best lambda: {}".format(best_lambda))
        return best_lambda
    elif reg_type == "ridge":
        best_lambda = gsCV.best_params_["alpha"]    
        print("best lambda: {}".format(best_lambda))
        return best_lambda
    elif reg_type == "elasticnet":
        best_alpha   = gsCV.best_params_["alpha"]
        best_l1ratio = gsCV.best_params_["l1_ratio"]
        best_lambda_1 = best_alpha * 2 * data.shape[0] * best_l1ratio
        best_lambda_2 = best_alpha * data.shape[0] * (1-best_l1ratio)
        print("best lambda1: {}".format( best_lambda_1 ))
        print("best lambda2: {}".format( best_lambda_2 ))
        return best_lambda_1, best_lambda_2

# ------------------------------------------------------------------------------------------------------
# Best Subset Regression
# ------------------------------------------------------------------------------------------------------
def best_subset_regression(dep_var, exp_vars, max_vars=3):
    '''
    Best Subset Regression 
    '''
    def best_subset(x, y, l_0):
        # Mixed Integer Programming in feature selection
        M = 1000
        n_factor = x.shape[1]
        z = cp.Variable(n_factor, boolean=True)
        beta = cp.Variable(n_factor)
        alpha = cp.Variable(1)

        def MIP_obj(x,y,b,a):
            return cp.norm(y-cp.matmul(x,b)-a,2)

        best_subset_prob = cp.Problem(
            objective   = cp.Minimize( MIP_obj(x, y, beta, alpha) ), 
            constraints = [cp.sum(z)<=l_0, beta+M*z>=0, M*z>=beta]
        )
        best_subset_prob.solve()
        return beta.value, alpha.value
    
    # perform best subset regression
    betas, alpha = best_subset(exp_vars, dep_var, max_vars)
    betas[np.abs(betas) <= 1e-7] = 0.0
    
    return betas, alpha

def display_betas(betas, names):
    '''
    Simply returns the betas coefficients (from a linear regression model) in a pd.DataFrame.
    The input betas vector has to be a list or a np.array
    '''
    return pd.DataFrame(betas, columns=["beta"], index=names+["alpha"]).T






def regime_hist(asset_rets, regime):
    '''
    Plot the histogram return of normal and crash regime givne in input a pd.Series of returns 
    and the associated vector regime
    '''
    # compute returns of normal and crash regimes
    asset_rets_g = asset_rets[regime==1]
    asset_rets_c = asset_rets[regime==-1]
    # plot
    fig, ax = plt.subplots(1,2,figsize=(14,5))
    asset_rets_g.hist(ax=ax[0], bins=25, grid=True, color='green', alpha=0.4, density=False,  label=None)
    asset_rets_c.hist(ax=ax[1], bins=25, grid=True, color='red', alpha=0.4, density=False, label=None)
    ax[0].axvline(x=asset_rets_g.mean(), linestyle="--", color="green", label="mean: {:.2f}".format(asset_rets_g.mean()))
    ax[0].axvline(x=asset_rets_g.median(), linestyle="--", color="blue", label="median: {:.2f}".format(asset_rets_g.median()))
    ax[1].axvline(x=asset_rets_c.mean(), linestyle="--", color="red", label="mean: {:.2f}".format(asset_rets_c.mean()))
    ax[1].axvline(x=asset_rets_c.median(), linestyle="--", color="blue", label="median: {:.2f}".format(asset_rets_c.median()))
    ax[0].set_xlabel('returns')
    ax[1].set_xlabel('returns')
    ax[0].set_ylabel('frequency')
    ax[1].set_ylabel('frequency')
    ax[0].set_title("Histogram returns of {} under Normal regime".format(asset_rets.name))
    ax[1].set_title("Histogram returns of {} under Crash regime".format(asset_rets.name))
    ax[0].legend()
    ax[1].legend()
    plt.show()
    return ax

def qqplot(rets, linetype="r"):
    '''
    Quantile-Quantile plot of an input pd.Series or np.array of returns using 
    statsmodels qqplot method. The variable linetype can be = "45","s","r",or "q". 
    '''
    fig, ax = plt.subplots(1,1,figsize=(7,5))    
    sm.qqplot(rets.values, line=linetype, ax=ax)
    plt.title("Q-Q Plot of {}".format(rets.name))
    plt.ylabel('returns')
    plt.grid()
    plt.show()
    
def regime_plot_cdf(ret_g, ret_c):
    '''
    Plot of the Cumulative Distribution Functions of a normal regime returns 
    and a crash regime returns pd.Series.
    '''
    xg, yg = ecdf(ret_g)
    xc, yc = ecdf(ret_c)
    fig, ax = plt.subplots(1,1,figsize=(7,5))
    ax.plot(xg, yg, color='green', label='Normal regime')
    ax.plot(xc, yc, color='red', label='Crash regime')
    ax.set_xlabel('returns')
    ax.set_ylabel('cumulative probability')
    ax.set_title("Cumulative Density of {}".format(ret_g.name))
    ax.grid()
    ax.legend()
    plt.show()
    return ax
    
def ecdf(data):
    '''
    Compute ECDF for a one-dimensional array of measurements.
    '''
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y



def trend_filtering(data,lambda_value):
    '''
    Runs the trend-filtering algorithm to separate regimes in a given series of returns.
    The input data has to be a np.array of returns and lambda is a constant scalar.
    '''
    # objective function of the trend-filtering algorithm
    def trend_filtering_obj(x,beta,lambd):
        return cp.norm(x-beta,2)**2 + lambd*cp.norm(cp.matmul(D, beta),1)

    n = np.size(data)
    x_ret = data.reshape(n)

    # creating first-order derivatives matrix
    Dfull = np.diag([1]*n) - np.diag([1]*(n-1),1)
    D = Dfull[0:(n-1),]

    # set up the proble
    beta  = cp.Variable(n)
    lambd = cp.Parameter(nonneg=True)
    problem = cp.Problem( cp.Minimize(trend_filtering_obj(x_ret, beta, lambd)) )

    # solve the problem
    lambd.value = lambda_value
    problem.solve()

    return beta.value

def regime_switch(betas,threshold=1e-5):
    '''
    Returns a list of starting points of each regime given in input 
    the beta vector as output from a trend-filtering algorithm.
    '''
    n = len(betas)
    init_points = [0]
    curr_reg = (betas[0]>threshold)
    for i in range(n):
        if (betas[i]>threshold) == (not curr_reg):
            curr_reg = not curr_reg
            init_points.append(i)
    init_points.append(n)
    return init_points

def trend_filtering_plot(rets, lambda_value=0.1, figx=10, figy=5):
    '''
    Return a plot of the original series and the filtered fitted series given 
    an input lambda value. 
    '''
    # finding betas by solving the minimization problems
    betas = trend_filtering(rets.values, lambda_value)
    betas = pd.Series(betas, index=rets.index)
    # plot
    fig, ax = plt.subplots(1,1,figsize=(figx,figy))
    rets.plot(ax=ax, grid=True, alpha=0.4, label='original returns')
    betas.plot(ax=ax, grid=True, label='fitted series')
    ax.set_ylabel('returns')
    ax.legend()
    return ax

def plot_regime_color(data, lambda_value=0.1, figx=9, figy=6):
    '''
    Plot a timeseries and corresponding regimes (normal or crash) according to 
    the lambda value in input. Regimes are indetified by vertical coloured rectangles.
    The input lambda_value is the lambda parameter (scalar) used by the 
    trend-filtering algorithm.
    '''
    # compute returns
    rets = erk.compute_returns(data).dropna()
    # get betas from trend-filtering 
    betas = trend_filtering(rets.values, lambda_value)
    # find regimes switching points
    regimelist = regime_switch(betas)
    curr_reg = np.sign(betas[0]-1e-5)
    
    fig, ax = plt.subplots(1,1,figsize=(figx,figy))
    for i in range(len(regimelist)-1):
        if curr_reg == 1:
            pass
            # uncomment below if we want to color the normal regimes
            #ax.axhspan(0, data.max(), xmin=regimelist[i]/regimelist[-1], xmax=regimelist[i+1]/regimelist[-1], 
            #          facecolor="green", alpha=0.3)
        else:
            ax.axhspan(0, data.max(),  xmin=regimelist[i]/regimelist[-1], xmax=regimelist[i+1]/regimelist[-1], 
                       facecolor='gray', alpha=0.5)
        curr_reg = -1 * curr_reg
        
    data.plot(ax=ax, grid=True)
    ax.set_ylabel("value")
    ax.set_title('Regime plot')
    ax.set_yscale('log')
    plt.show()
    return ax

def efficient_frontier_two_regimes(rets, rets_g, rets_c, periods_per_year=12, n_ports=20, n_scenarios=10000, figx=8, figy=6):
    '''
    Plot the efficient frontiers from single regime model (i.e., traditional Markowitz efficient frontier) versus
    the efficient frontier obtained by sampling separated returns for the normal and the crash regime that given input. 
    The variable n_ports denotes the number of points on the efficient frontiers to be plotted, 
    whereas n_scenarios the total number of returns (normal+chrash) to be sampled for the wo-regimes case.  
    '''
    # annualize returns and computes the cov matrix 
    ann_rets = erk.annualize_rets(rets, periods_per_year=periods_per_year)
    cov_rets = rets.cov()
    
    # range of total expected returns 
    rets_range = np.linspace(ann_rets.min(),ann_rets.max(),n_ports)    
    
    # compute the efficient frontiers for the entire dataframe: single regime 
    vol_single, ret_single = [], []
    for r in rets_range:
        ww = erk.minimize_volatility(ann_rets, cov_rets, target_return=r)
        ret_single.append( erk.portfolio_return(ww, ann_rets) )
        vol_single.append( erk.annualize_vol( erk.portfolio_volatility(ww,cov_rets), periods_per_year=periods_per_year) )

    # compute the efficient frontiers for the normal and crash dataframes: two regimes 
    # sample from multivariate normal distribution with given mean and cov 
    
    # annual returns
    ann_rets_g = erk.annualize_rets(rets_g, periods_per_year=periods_per_year)
    ann_rets_c = erk.annualize_rets(rets_c, periods_per_year=periods_per_year)
    
    # sampling
    n_g = int( n_scenarios * rets_g.shape[0] / rets.shape[0] )
    s_1 = np.random.multivariate_normal(ann_rets_g, rets_g.cov()*periods_per_year, n_g)
    s_2 = np.random.multivariate_normal(ann_rets_c, rets_c.cov()*periods_per_year, n_scenarios-n_g)
    # note that we have multiplied the cov * periods_per_year because we are using annualized data, i.e., 
    # the mean value are the annual returns, hence the cov matrix has to be scaled accordingly 
    rets_two = pd.DataFrame( np.vstack((s_1,s_2)), columns=rets.columns )
    # new annual return and covariance matrix 
    ann_rets_two = rets_two.mean()
    cov_rets_two = rets_two.cov()
    # and now note that ann_rets_two is the mean since we sampled using means equal to the annual returns, 
    # then we do not have to annualize again
        
    vol_two, ret_two = [], []
    for r in rets_range:
        ww = erk.minimize_volatility(ann_rets_two, cov_rets_two, target_return=r)
        ret_two.append( erk.portfolio_return(ww, ann_rets_two) )
        vol_two.append( erk.portfolio_volatility(ww,cov_rets_two) )
        # here note that we don't have to annualize the volatility since the covmatrix was already annualized 
        
    fig, ax = plt.subplots(1,1,figsize=(figx,figy))
    ax.plot(vol_single, ret_single, "xb-", label="Eff. Frontier Single regime")
    ax.plot(vol_two, ret_two, "xr-", label="Eff. Frontier Two regimes")
    ax.grid()
    ax.legend()
    plt.show()
  
    return ax    

def regime_asset(n,mu1,mu2,Q1,Q2,p1,p2):
    '''
    Simulates normal and crash returns (a total of n) from multivariate distribution with given input means 
    and covariances and generate a regime vector by switching returns according to 
    probability transitions:
    - p1: probability of remaining in regime 1 if we are in regime 1 
    - p2: probability of going to regime 1 if we are in regime 2
    '''
    s_1 = np.random.multivariate_normal(mu1, Q1, n).T
    s_2 = np.random.multivariate_normal(mu2, Q2, n).T
    regime = np.ones(n)
    for i in range(n-1):
        if regime[i] == 1:
            if np.random.rand() > p1:
                regime[i+1] = 0
        else:
            if np.random.rand() < p2:
                regime[i+1] = 1
    return (regime*s_1 + (1-regime)*s_2).T

def transition_matrix(regime):
    '''
    Computes the transition matrix given the regime vector. Here
    - p11 is the probability of staying in regime 1 given that current regime is 1
    - p12 is the probability of moving to regime 2 given that current regime is 1
    - p21 is the probability of moving to regime 1 given that current regime is 2
    - p22 is the probability of staying in regime 2 given that current regime is 2
    Note that in the regime vector, regime 1 is 1 (supposed to be normal) 
    and regime 2 is -1 (supposed to be crash). 
    '''
    n1,n2,n3,n4 = 0,0,0,0
    for i in range(len(regime)-1):
        if regime[i] == 1:
            # current regime is 1
            if regime[i+1] == 1:
                n1 += 1
            else:
                n2 += 1
        else:
            # current regime is 0
            if regime[i+1] == 1:
                n3 += 1
            else:
                n4 += 1
    p11 = n1 / (n1+n2)
    p12 = n2 / (n1+n2)
    p21 = n3 / (n3+n4)
    p22 = n4 / (n3+n4)
    return p11, p12, p21, p22

def regime_based_simulated_rets(rets, rets_g, rets_c, regime, periods_per_years=12, n_years=50, n_scenarios=1000):
    '''
    Simulates regime-based returns given in input pd.Dataframe data and a regime vector 
    identified by regime_name (which is a column of data).
    The variable assets_name denotes the columns' name of data corresponding to the assets we want to use.
    The simulated results are stored into a (n_year*periods_per_years)*len(assets_name)*n_scenario tensor 
    '''
    # compute returns
    #rets_all = erk.compute_returns( data[assets_name] ).dropna()
    
    # returns of regular and crash regime according to the regime vector 
    #rets_g = rets_all[ (data[regime_name]==1)[1:]  ]
    #rets_c = rets_all[ (data[regime_name]==-1)[1:] ]
    
    # compute transition probabilities given the input regime vector 
    p1, _, p2, _ = transition_matrix( regime )
    
    # compute fixed return period (like return per month)
    mu1 = ( (1+rets_g).prod() )**(1/rets_g.shape[0]) - 1
    mu2 = ( (1+rets_c).prod() )**(1/rets_c.shape[0]) - 1
    # covariances
    Q1 = rets_g.cov()
    Q2 = rets_c.cov()
    
    r_all = np.zeros((n_years*periods_per_years, len(rets.columns), n_scenarios))
    for i in range(n_scenarios):
        r_all[:,:,i] = regime_asset(n_years*periods_per_years,mu1,mu2,Q1,Q2,p1,p2)
    
    return r_all

def simulate_fund_wealth(r_all, assets_names, holdings, start=100):
    '''
    Generates the portfolio simulated wealth given in input a r_all array of many scenarios 
    for a given array of (fixed) weights holdings
    '''
    n_scenarios = r_all.shape[2]

    # create weights dataframe
    ww = pd.DataFrame( [holdings]*r_all.shape[0], columns=assets_names)

    portf_sim = pd.DataFrame()
    portfolio_rets = pd.DataFrame()
    for n in range(n_scenarios):
        sim_rets = pd.DataFrame(r_all[:,:,n], columns=assets_names)
        portfolio_wealth = start * (1 + ww.multiply(sim_rets).sum(axis=1) ).cumprod()
        portf_sim = pd.concat([portf_sim,portfolio_wealth], axis=1)

    portf_sim.columns = range(r_all.shape[2])
    return portf_sim
    