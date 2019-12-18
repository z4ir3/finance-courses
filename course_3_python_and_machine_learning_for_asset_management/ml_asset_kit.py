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