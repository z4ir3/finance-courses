# Factor-Model Library
import time
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
from scipy.optimize import minimize

def plot_returns(data, names, flag='Total Return', date='Date', printFinalVals = False):
    '''
    Returns a plot of the returns.
    INPUTS:
        names: string, name of column to be plotted, or list, in which case it plots all of them
        data: pd dataframe, where the data is housed
        flag: string, Either Total Return or Monthly Return
        date: string, column name corresponding to the date variable
        printFinalVals: Boolean, if True, prints the final Total Return
    Outputs:
        a plot
    '''
    #Clean Inputs:
    if(date not in data.columns):
        print ('date column not in the pandas df')
        return
    if(type(names) is str):
        names = [names]
    for name in names:
        if(name not in data.columns):
            print ('column ' + name + ' not in pandas df')
            return
    #If the inputs are clean, create the plot
    data = data.sort_values(date).copy()
    data.reset_index(drop=True, inplace=True)
    data[date] = pd.to_datetime(data[date])

    if (flag == 'Total Return'):
        n = data.shape[0]
        totalReturns = np.zeros((n,len(names)))
        totalReturns[0,:] = 1.
        for i in range(1,n):
            totalReturns[i,:] = np.multiply(totalReturns[i-1,:], (1+data[names].values[i,:]))
        for j in range(len(names)):
            plt.semilogy(data[date], totalReturns[:,j])

        plt.title('Total Return Over Time')
        plt.ylabel('Total Return')
        plt.legend(names)
        plt.xlabel('Date')
        plt.show()
        if(printFinalVals):
            print(totalReturns[-1])
    elif (flag == 'Relative Return'):
        for i in range(len(names)):
            plt.plot(data[date], data[names[i]])
        plt.title('Returns Over Time')
        plt.ylabel('Returns')
        plt.legend(names)
        plt.xlabel('Date')
        plt.show()
    else:
        print ('flag variable must be either Total Return or Monthly Return')


#Helper Functions
def create_options():
    '''create standard options dictionary to be used as input to regression functions'''
    options = dict()
    options['timeperiod'] = 'all'
    options['date'] = 'Date'
    options['returnModel'] = False
    options['printLoadings'] = True
    return options

def create_options_lasso():
    options = create_options()
    options['lambda'] = 1
    return options

def create_options_ridge():
    options = create_options()
    options['lambda'] = 1
    return options

def create_options_cv_lasso():
    options = create_options()
    options['maxLambda'] = .25
    options['nLambdas'] = 100
    options['randomState'] = 7777
    options['nFolds'] = 10
    return options

def create_options_cv_ridge():
    options = create_options()
    options['maxLambda'] = .25
    options['nlambdas'] = 100
    options['randomState'] = 7777
    options['nFolds'] = 10
    return options

def create_options_cv_elastic_net():
    options = create_options()
    options['maxLambda'] = .25
    options['maxL1Ratio'] = .99
    options['nLambdas'] = 50
    options['nL1Ratios'] = 50
    options['randomState'] = 7777
    options['nFolds'] = 10
    return options

def create_options_best_subset():
    '''create standard options dictionary to be used as input to regression functions'''
    options = create_options()
    options['returnModel'] = False
    options['printLoadings'] = True
    options['maxVars'] = 3
    return options

def create_options_relaxed_lasso(CV=True, lambda1=.25):
    options = create_options()
    options['CV'] = CV
    options['gamma'] = .5
    if(CV):
        options['maxLambda'] = .25
        options['nLambdas'] = 100
        options['randomState'] = 7777
        options['nFolds'] = 10
    else:
        #Need to specify lambda1
        options['lambda'] = .25
    return options


def create_dictionary_for_analysis(method, methodDict=None):
    '''create_dictionary_for_anlsis creates the options dictionary that can be used as an input to a factor model
    INPUTS:
        method: string, defines the method
    OUTPUTS:
        methodDict: dictionary, keys are specific options the user wants to specify, values are the values of those options
    '''
    if(method == 'OLS'):
        options = create_options()
    elif(method == 'CVLasso'):
        options = create_options_cv_lasso()
    elif(method == 'CVRidge'):
        options = create_options_cv_ridge()
    elif(method == 'CVElasticNet'):
        options = create_options_cv_elastic_net()
    elif(method == 'BestSubset'):
        options = create_options_best_subset()
    elif(method == 'RelaxedLasso'):
        options = create_options_relaxed_lasso()
    else:
        print('Bad Method Specification for Train')
        return
    options['returnModel'] = True
    options['printLoadings'] = False
    options['date'] = 'DataDate'
    for key in methodDict:
        options[key] = methodDict[key]
    return options



def print_timeperiod(data, dependentVar, options):
    '''print_timeperiod takes a a dependent varaible and a options dictionary, prints out the time period
    INPUTS:
        data: pandas df, df with the data
        dependentVar: string, name of dependent variable
        options: dictionary, should constain at least two elements, timeperiod, and date
            timeperiod: string, if == all, means use entire dataframe, otherwise filter the df on this value
            date: name of datecol
    OUTPUTS:
        printed stuff
    '''
    print ('Dependent Variable is ' + dependentVar)
    if(options['timeperiod'] == 'all'):
        sortedValues = data.sort_values(options['date'])[options['date']].reset_index(drop=True)
        n = sortedValues.shape[0]
        beginDate = sortedValues[0]
        endDate = sortedValues[n-1]
        print ('Time period is between ' + num_to_month(beginDate.month) +  ' ' + str(beginDate.year) + ' to ' + num_to_month(endDate.month) +  ' ' + str(endDate.year) + ' inclusive   ')        
    else:
        print ('Time period is ' + options['timeperiod'])

def display_factor_loadings(intercept, coefs, factorNames, options):
    '''display_factor_loadings takes an intercept, coefs, factorNames and options dict, and prints the factor loadings in a readable way
    INPUTS:
        intercept: float, intercept value
        coefs: np array, coeficients from pandas df
        factorNames: list, names of the factors
        options: dict, should contain at least one key, nameOfReg
            nameOfReg: string, name for the regression
    Outputs:
        output is printed
    '''
    loadings = np.insert(coefs, 0, intercept)
    if('nameOfReg' not in options.keys()):
        name = 'No Name'
    else:
        name = options['nameOfReg']
    out = pd.DataFrame(loadings, columns=[name])
    out = out.transpose()
    fullNames = ['Intercept'] + factorNames
    out.columns = fullNames
    print(out)

def best_subset(x,y,l_0):
    # Mixed Integer Programming in feature selection
    M = 1000
    n_factor = x.shape[1]
    z = cp.Variable(n_factor, boolean=True)
    beta = cp.Variable(n_factor)
    alpha = cp.Variable(1)

    def MIP_obj(x,y,b,a):
        return cp.norm(y-cp.matmul(x,b)-a,2)

    best_subset_prob = cp.Problem(cp.Minimize(MIP_obj(x, y, beta, alpha)), 
                             [cp.sum(z)<=l_0, beta+M*z>=0, M*z>=beta])
    best_subset_prob.solve()
    return alpha.value, beta.value


#First function, linear factor model build
def linear_regression(data, dependentVar, factorNames, options):
    '''linear_regression takes in a dataset and returns the factor loadings using least squares regression
    INPUTS:
        data: pandas df, data matrix, should constain the date column and all of the factorNames columns
        dependentVar: string, name of dependent variable
        factorNames: list, elements should be strings, names of the independent variables
        options: dictionary, should constain at least two elements, timeperiod, and date
            timeperiod: string, if == all, means use entire dataframe, otherwise filter the df on this value
            date: name of datecol
            returnModel: boolean, if true, returns model
    Outputs:
        reg: regression object from sikitlearn
        also prints what was desired
    '''
    #first filter down to the time period
    if(options['timeperiod'] == 'all'):
        newData = data.copy()
    else:
        newData = data.copy()
        newData = newData.query(options['timeperiod'])

    #perform linear regression
    linReg = LinearRegression(fit_intercept=True)
    linReg.fit(newData[factorNames], newData[dependentVar])
    
    if (options['printLoadings'] == True):
        #Now print the results
        print_timeperiod(newData, dependentVar, options)
        # Now print the factor loadings
        display_factor_loadings(linReg.intercept_, linReg.coef_, factorNames, options)

    if(options['returnModel']):
        return linReg


def lasso_regression(data, dependentVar, factorNames, options):
    '''lasso_regression takes in a dataset and returns the factor loadings using lasso regression
    INPUTS:
        data: pandas df, data matrix, should constain the date column and all of the factorNames columns
        dependentVar: string, name of dependent variable
        factorNames: list, elements should be strings, names of the independent variables
        options: dictionary, should constain at least two elements, timeperiod, and date
            timeperiod: string, if == all, means use entire dataframe, otherwise filter the df on this value
            printLoadings: boolean, if true, prints the coeficients

            date: name of datecol
            returnModel: boolean, if true, returns model
            alpha: float, alpha value for LASSO regression
            NOTE: SKLearn calles Lambda Alpha.  Also, it uses a scaled version of LASSO argument, so here I scale when converting lambda to alpha
    Outputs:
        reg: regression object from sikitlearn
        also prints what was desired
    '''
    if('lambda' not in options.keys()):
        print ('lambda not specified in options')
        return

    #first filter down to the time period
    if(options['timeperiod'] == 'all'):
        newData = data.copy()
    else:
        newData = data.copy()
        newData = newData.query(options['timeperiod'])

    #perform linear regression
    lassoReg = Lasso(alpha=options['lambda']/(2*data.shape[0]), fit_intercept=True)
    lassoReg.fit(newData[factorNames], newData[dependentVar])
    
    if (options['printLoadings'] == True):
        #Now print the results
        print_timeperiod(newData, dependentVar, options)
        print('lambda = ' + str(options['lambda']))

        #Now print the factor loadings
        display_factor_loadings(lassoReg.intercept_, lassoReg.coef_, factorNames, options)

    if(options['returnModel']):
        return lassoReg

def ridge_regression(data, dependentVar, factorNames, options):
    '''ridge_regression takes in a dataset and returns the factor loadings using ridge regression
    INPUTS:
        data: pandas df, data matrix, should constain the date column and all of the factorNames columns
        dependentVar: string, name of dependent variable
        factorNames: list, elements should be strings, names of the independent variables
        options: dictionary, should constain at least two elements, timeperiod, and date
            timeperiod: string, if == all, means use entire dataframe, otherwise filter the df on this value
            date: name of datecol
            returnModel: boolean, if true, returns model
            lambda: float, alpha value for Ridge regression
            printLoadings: boolean, if true, prints the coeficients
    Outputs:
        reg: regression object from sikitlearn
        also prints what was desired
    '''
    if('lambda' not in options.keys()):
        print ('lambda not specified in options')
        return

    #first filter down to the time period
    if(options['timeperiod'] == 'all'):
        newData = data.copy()
    else:
        newData = data.copy()
        newData = newData.query(options['timeperiod'])

    #perform linear regression
    ridgeReg = Ridge(alpha=options['lambda'], fit_intercept=True)
    ridgeReg.fit(newData[factorNames], newData[dependentVar])
    
    if (options['printLoadings'] == True):
        #Now print the results
        print_timeperiod(newData, dependentVar, options)
        print('lambda = ' + str(options['lambda']))

        #Now print the factor loadings
        display_factor_loadings(lassoReg.intercept_, lassoReg.coef_, factorNames, options)

    if(options['returnModel']):
        return ridgeReg

def best_subset_regression(data, dependentVar, factorNames, options):
    '''best_subset_regression takes in a dataset and returns the factor loadings using best subset regression
    INPUTS:
        data: pandas df, data matrix, should constain the date column and all of the factorNames columns
        dependentVar: string, name of dependent variable
        factorNames: list, elements should be strings, names of the independent variables
        options: dictionary, should constain at least two elements, timeperiod, and date
            timeperiod: string, if == all, means use entire dataframe, otherwise filter the df on this value
            date: name of datecol
            returnModel: boolean, if true, returns model
            maxVars: int, maximum number of factors that can have a non zero loading in the resulting regression
            printLoadings: boolean, if true, prints the coeficients
    Outputs:
        reg: regression object from sikitlearn
        also prints what was desired
    '''
    #Check dictionary for maxVars option
    if('maxVars' not in options.keys()):
        print ('maxVars not specified in options')
        return

    if(options['timeperiod'] == 'all'):
        newData = data.copy()
    else:
        newData = data.copy()
        newData = newData.query(options['timeperiod'])

    #perform linear regression
    alpha, beta = best_subset(data[factorNames].values, data[dependentVar].values, options['maxVars'])
    #round beta values to zero
    beta[np.abs(beta) <= 1e-7] = 0.0
    
    if (options['printLoadings'] == True):
        #Now print the results
        print_timeperiod(newData, dependentVar, options)
        print('Max Number of Non-Zero Variables is ' + str(options['maxVars']))

        #Now print the factor loadings
        display_factor_loadings(alpha, beta, factorNames, options)

    if(options['returnModel']):
        out = LinearRegression()
        out.intercept_ = alpha[0]
        out.coef_ = beta
        return out

def cross_validated_lasso_regression(data, dependentVar, factorNames, options):
    '''cross_validated_lasso_regression takes in a dataset and returns the factor loadings using lasso regression and cross validating the choice of lambda
    INPUTS:
        data: pandas df, data matrix, should constain the date column and all of the factorNames columns
        dependentVar: string, name of dependent variable
        factorNames: list, elements should be strings, names of the independent variables
        options: dictionary, should constain at least two elements, timeperiod, and date
            timeperiod: string, if == all, means use entire dataframe, otherwise filter the df on this value
            date: name of datecol
            returnModel: boolean, if true, returns model
            printLoadings: boolean, if true, prints the coeficients

            maxLambda: float, max lambda value passed
            nLambdas: int, number of lambda values to try
            randomState: integer, sets random state seed
            nFolds: number of folds
            NOTE: SKLearn calles Lambda Alpha.  Also, it uses a scaled version of LASSO argument, so here I scale when converting lambda to alpha
    Outputs:
        reg: regression object from sikitlearn
        also prints what was desired
    '''
    #Test timeperiod
    if(options['timeperiod'] == 'all'):
        newData = data.copy()
    else:
        newData = data.copy()
        newData = newData.query(options['timeperiod'])

    #Do CV Lasso
    alphaMax = options['maxLambda'] / (2*newData.shape[0])
    print('alphaMax = ' + str(alphaMax))
    alphas = np.linspace(1e-6, alphaMax, options['nLambdas'])
    if(options['randomState'] == 'none'):
        lassoTest = Lasso(fit_intercept=True)
    else:
        lassoTest = Lasso(random_state = options['randomState'], fit_intercept=True)

    tuned_parameters = [{'alpha': alphas}]

    clf = GridSearchCV(lassoTest, tuned_parameters, cv=options['nFolds'], refit=True)
    clf.fit(newData[factorNames],newData[dependentVar])
    lassoBest = clf.best_estimator_
    alphaBest = clf.best_params_['alpha']
    print('Best Alpha')
    print(clf.best_params_['alpha'])


    if (options['printLoadings'] == True):
        #Now print the results
        print_timeperiod(newData, dependentVar, options)
        print('best lambda = ' + str(alphaBest*2*newData.shape[0]))
        #Now print the factor loadings
        display_factor_loadings(lassoBest.intercept_, lassoBest.coef_, factorNames, options)

    if(options['returnModel']):
        return lassoBest

def cross_validated_ridge_regression(data, dependentVar, factorNames, options):
    '''cross_validated_ridge_regression takes in a dataset and returns the factor loadings using ridge regression and choosing lambda via ridge regression
    INPUTS:
        data: pandas df, data matrix, should constain the date column and all of the factorNames columns
        dependentVar: string, name of dependent variable
        factorNames: list, elements should be strings, names of the independent variables
        options: dictionary, should constain at least two elements, timeperiod, and date
            timeperiod: string, if == all, means use entire dataframe, otherwise filter the df on this value
            date: name of datecol
            returnModel: boolean, if true, returns model
            printLoadings: boolean, if true, prints the coeficients

            maxLambda: float, max lambda value passed
            nLambdas: int, number of lambda values to try
            randomState: integer, sets random state seed
            nFolds: number of folds
            NOTE: SKLearn calles Lambda Alpha.  So I change Lambda -> Alpha in the following code
    Outputs:
        reg: regression object from sikitlearn
        also prints what was desired
    '''
    #Test timeperiod
    if(options['timeperiod'] == 'all'):
        newData = data.copy()
    else:
        newData = data.copy()
        newData = newData.query(options['timeperiod'])

    #Do CV Lasso
    alphaMax = options['maxLambda']
    alphas = np.linspace(1e-6, alphaMax, options['nLambdas'])
    if(options['randomState'] == 'none'):
        ridgeTest = Ridge(fit_intercept=True)
    else:
        ridgeTest = Ridge(random_state = options['randomState'], fit_intercept=True)

    tuned_parameters = [{'alpha': alphas}]

    clf = GridSearchCV(ridgeTest, tuned_parameters, cv=options['nFolds'], refit=True)
    clf.fit(newData[factorNames],newData[dependentVar])
    ridgeBest = clf.best_estimator_
    alphaBest = clf.best_params_['alpha']

    if (options['printLoadings'] == True):
        #Now print the results
        print_timeperiod(newData, dependentVar, options)
        print('best alpha = ' + str(alphaBest))
        #Now print the factor loadings
        display_factor_loadings(ridgeBest.intercept_, ridgeBest.coef_, factorNames, options)

    if(options['returnModel']):
        return ridgeBest

def cross_validated_elastic_net_regression(data, dependentVar, factorNames, options):
    '''cross_validated_elastic_net_regression takes in a dataset and returns the factor loadings using elastic net, also chooses alpha and l1 ratio via cross validation
    INPUTS:
        data: pandas df, data matrix, should constain the date column and all of the factorNames columns
        dependentVar: string, name of dependent variable
        factorNames: list, elements should be strings, names of the independent variables
        options: dictionary, should constain at least two elements, timeperiod, and date
            timeperiod: string, if == all, means use entire dataframe, otherwise filter the df on this value
            date: name of datecol
            returnModel: boolean, if true, returns model
            printLoadings: boolean, if true, prints the coeficients

            maxLambda: float, max lambda value passed
            nLambdas: int, number of lambda values to try
            maxL1Ratio: float
            randomState: integer, sets random state seed
            nFolds: number of folds
            NOTE: SKLearn calles Lambda Alpha.  So I change Lambda -> Alpha in the following code
    Outputs:
        reg: regression object from sikitlearn
        also prints what was desired
    '''
    #Test timeperiod
    if(options['timeperiod'] == 'all'):
        newData = data.copy()
    else:
        newData = data.copy()
        newData = newData.query(options['timeperiod'])

    #Do CV Lasso
    alphaMax = options['maxLambda']/(2*newData.shape[0])
    alphas = np.linspace(1e-6, alphaMax, options['nLambdas'])
    l1RatioMax = options['maxL1Ratio']
    l1Ratios = np.linspace(1e-6, l1RatioMax, options['nL1Ratios'])
    if(options['randomState'] == 'none'):
        elasticNetTest = ElasticNet(fit_intercept=True)
    else:
        elasticNetTest = ElasticNet(random_state = options['randomState'], fit_intercept=True)

    tuned_parameters = [{'alpha': alphas, 'l1_ratio': l1Ratios}]

    clf = GridSearchCV(elasticNetTest, tuned_parameters, cv=options['nFolds'], refit=True)
    clf.fit(newData[factorNames],newData[dependentVar])
    elasticNetBest = clf.best_estimator_
    alphaBest = clf.best_params_['alpha']
    l1RatioBest = clf.best_params_['l1_ratio']

    if (options['printLoadings'] == True):
        #Now print the results
        print_timeperiod(newData, dependentVar, options)
        print('best lambda1 = ' + str(alphaBest*2*newData.shape[0]*l1RatioBest))
        print('best lambda2 = ' + str(newData.shape[0]*alphaBest*(1-l1RatioBest)))
        #Now print the factor loadings
        display_factor_loadings(elasticNetBest.intercept_, elasticNetBest.coef_, factorNames, options)

    if(options['returnModel']):
        return elasticNetBest


def relaxed_lasso_regression(data, dependentVar, factorNames, options):
    '''cross_validated_lasso_regression takes in a dataset and returns the factor loadings using lasso regression and cross validating the choice of lambda
    INPUTS:
        data: pandas df, data matrix, should constain the date column and all of the factorNames columns
        dependentVar: string, name of dependent variable
        factorNames: list, elements should be strings, names of the independent variables
        options: dictionary, should constain at least two elements, timeperiod, and date
            timeperiod: string, if == all, means use entire dataframe, otherwise filter the df on this value
            date: name of datecol
            returnModel: boolean, if true, returns model
            printLoadings: boolean, if true, prints the coeficients

            maxLambda: float, max lambda value passed
            nLambdas: int, number of lambda values to try
            randomState: integer, sets random state seed
            nFolds: number of folds
            NOTE: SKLearn calles Lambda Alpha.  Also, it uses a scaled version of LASSO argument, so here I scale when converting lambda to alpha
    Outputs:
        reg: regression object from sikitlearn
        also prints what was desired
    '''
    #Step 1: Build the LASSO regression
    #Check if you cross validate
    optionsNew = options
    optionsNew['printLoadings'] = False
    if(options['CV']):
        lassoModel = cross_validated_lasso_regression(data, dependentVar, factorNames, optionsNew)
    else:
        lassoModel = lasso_regression(data, dependentVar, factorNames, optionsNew)

    #Step 2: Extract Non-Zero Factor Loadings
    listNonZeroLoadings = []
    for i in range(len(factorNames)):
        if (lassoModel.coef_[i] != 0):
            listNonZeroLoadings.append(factorNames[i])

    #Step 2a: Check if the lambda value chosen is too large
    if(len(listNonZeroLoadings) == 0):
        print('Lambda Value Set To Big, Model is Null Model')
        reg = LinearRegression()
        reg.coef_ = np.zeros((len(factorNames),))
        reg.intercept_ = 0.0
        return reg
    
    #Step 3: Run OLS using just the non-zero coeficients from the LASSO regression
    olsReg = linear_regression(data, dependentVar, listNonZeroLoadings, optionsNew)

    #Step 4: Average the two models together
    coefs = np.zeros((len(factorNames),))
    for i in range(len(factorNames)):
        if(lassoModel.coef_[i] != 0):
            ind = listNonZeroLoadings.index(factorNames[i])
            coefs[i] = optionsNew['gamma']*lassoModel.coef_[i] + (1-optionsNew['gamma'])*olsReg.coef_[ind]

    reg = LinearRegression()
    reg.coef_ = coefs
    reg.intercept_ = optionsNew['gamma']*lassoModel.intercept_ + (1-optionsNew['gamma'])*olsReg.intercept_

    if (options['printLoadings'] == True):
        #Now print the results
        if(options['timeperiod'] == 'all'):
            newData = data.copy()
        else:
            newData = data.copy()
            newData = newData.query(options['timeperiod'])

        print_timeperiod(newData, dependentVar, options)
        if(options['CV']):
            print('best lambda = ' + str(lassoModel.alpha*2*newData.shape[0]))
        else:
            print('lambda = ' + str(options['lambda']))
        #Now print the factor loadings
        display_factor_loadings(reg.intercept_, reg.coef_, factorNames, options)

    return reg


def run_factor_model(data, dependentVar, factorNames, method, options):
    '''run_Factor_model allows you to specify the method to create a model, returns a model object according to the method you chose
    INPUTS:
        data: pandas df, must contain the columns specified in factorNames and dependentVar
        dependentVar: string, dependent variable
        factorNames: list of strings, names of independent variables
        method: string, name of method to be used.  Supports OLS, LASSO, CVLASSO atm
        options: dictionary object, controls the hyperparameters of the method
    Outputs:
        out: model object'''

    #Make sure the options dictionary has the correct settings
    options['returnModel'] = True
    options['printLoadings'] = False

    #Now create the appropriate model
    if (method == 'OLS'): #run linear model
        return linear_regression(data, dependentVar, factorNames, options)
    if (method == 'LASSO'):
        return lasso_regression(data, dependentVar, factorNames, options)
    if (method == 'Ridge'):
        return ridge_regression(data, dependentVar, factorNames, options)
    if (method == 'CVLasso'):
        return cross_validated_lasso_regression(data, dependentVar, factorNames, options)
    if (method == 'CVRidge'):
        return cross_validated_ridge_regression(data, dependentVar, factorNames, options)
    if (method == 'CVElasticNet'):
        return cross_validated_elastic_net_regression(data, dependentVar, factorNames, options)
    if (method == 'BestSubset'):
        return best_subset_regression(data, dependentVar, factorNames, options)
    if (method == 'RelaxedLasso'):
        return relaxed_lasso_regression(data, dependentVar, factorNames, options)
    else:
        print ('Method ' + method + ' not supported')

# Function to create a time series of factor loadings using a trailing window
def compute_trailing_factor_regressions(data, dependentVar, factorNames, window, method, options, dateCol='Date', printTime=False):
    '''compute_trailing_factor_regressions computes the factor regresssions using a trailing window, returns a pandas df object
    INPUTS:
        data: pandas df, must constain the columns dependentVar, and the set of columns factorNames
        dependentVar: string, names the dependent variable, must be a column in the dataframe data
        factorNames: list of string, elements must be members
        window: int, lookback window, measured in number of trading days
        method: string, can be OLS, LASSO or CVLasso
        options: dictionary, options dictionary
        dateCol: string, name of date column, also must be included in data
        printTime: boolean, if True, prints time it took to run the regressions
    Outputs:
        regressionValues: pandas df, rows should be different dates, columns should be factor loadings calculated using the trailing window
    '''
    if(printTime):
        start = time.time()
    options['returnModel'] = True
    options['printLoadings'] = False
    days = list(np.sort(data[dateCol].unique()))
    listOfFactorsAndDate = [dateCol] + factorNames
    regressionValues = pd.DataFrame(columns=listOfFactorsAndDate)
    for i in range(window, len(days)):
        #Filter the data
        filtered = data[(data[dateCol] <= days[i]) & (data[dateCol] >= days[i-window])]
        #Run the regression
        reg = run_factor_model(filtered, dependentVar, factorNames, method, options)
        #Append the regression values
        newRow = pd.DataFrame(reg.coef_)
        newRow = newRow.transpose()
        newRow.columns = factorNames
        newRow[dateCol] = days[i]
        regressionValues = regressionValues.append(newRow, sort=True)
    if(printTime):
        print('regression took ' + str((time.time() - start)/60.) + ' minutes')
    return regressionValues

#Asorted Nonsense
def num_to_month(month):
    #num to month returns the name of the month, input is an integer
    if (month==1):
        return 'January'
    if (month==2):
        return 'Febuary'
    if (month==3):
        return 'March'
    if (month==4):
        return 'April'
    if (month==5):
        return 'May'
    if (month==6):
        return 'June'
    if (month==7):
        return 'July'
    if (month==8):
        return 'August'
    if (month==9):
        return 'September'
    if (month==10):
        return 'October'
    if (month==11):
        return 'November'
    if (month==12):
        return 'December'


def data_time_periods(data, dateName):
    '''data_time_periods figures out if the data is daily, weekly, monthly, etc
    INPUTS:
        data: pandas df, has a date column in it with column name dateName
        dateName: string, name of column to be analysed
    '''
    secondToLast = data[dateName].tail(2)[:-1]
    last = data[dateName].tail(1)
    thingy = (last.values - secondToLast.values).astype('timedelta64[D]') / np.timedelta64(1, 'D')
    thingy = thingy[0]
    if (thingy > 200):
        return 'yearly'
    elif(thingy > 20):
        return 'monthly'
    elif(thingy > 5):
        return 'weekly'
    else:
        return 'daily'




















