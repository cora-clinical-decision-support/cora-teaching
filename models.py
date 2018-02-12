#!/usr/bin/python
# -*- coding: utf-8 -*-
# All the models

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import tree
from sklearn import preprocessing, metrics
import pandas as pd
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== PRE-PROCESSING ========== #

def encoding(nominal_col):
    """
    transform a column of categorical values to one of numeric values
    """

    le = preprocessing.LabelEncoder()
    le.fit(nominal_col)
    return le.transform(nominal_col)


def encoding_wrapper(X):
    for col_name in X.select_dtypes(include=['object']):
        X.loc[:,col_name] = encoding(X[col_name])
    return X


def filling_NAs(X):
    """
    fill in missing values with column means
    """

    fill = preprocessing.Imputer(strategy='mean')
    X_imputed = fill.fit_transform(X)

    return pd.DataFrame(X_imputed, columns = X.columns)

def floatFormatter(x):
    return '{0:.5f}'.format(float(x))

# ============== MODELS ============== #

def linReg(X_train, X_test, y_train, y_test, modelObj):
    """
    train a linear regression model
    """

    # get model parameters
    normalize = modelObj["parameters"][0]

    # train the classifier
    model = LinearRegression(normalize = normalize)
    model.fit(X=X_train, y=y_train)

    # predict
    y_pred = model.predict(X_test)

    # score the model
    coef = np.around(model.coef_, decimals=5).tolist()
    mse = metrics.mean_squared_error(y_test, y_pred) # 0 is perfect prediction
    var_score = metrics.r2_score(y_test, y_pred) # 1 is perfect prediction

    # format numbers '{0:.5f}'.format
    mse = floatFormatter(mse)
    var_score = floatFormatter(var_score)

    return (y_pred, coef, mse, var_score)


def treeClf(X, y, modelObj):
    """
    train a decision tree model
    """

    # get model parameters

    max_depth = float(modelObj['max_depth'])
    min_samples_leaf = modelObj['min_samples_leaf']

    # encode categorical features
    # ! must before dealing with NaNs

    X = encoding_wrapper(X)

    # deal with NaNs

    X = filling_NAs(X)
    
    # split train, test set

    (X_train, X_test, y_train, y_test) = train_test_split(X, y,
            test_size=0.3, random_state=17)

    # train the classifier

    model = tree.DecisionTreeClassifier(max_depth=max_depth,
            min_samples_leaf=min_samples_leaf)
    model.fit(X=X_train, y=y_train)
    
    return (model, X_test, y_test)


