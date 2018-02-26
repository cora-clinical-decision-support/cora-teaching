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
from collections import Counter
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import logging
from scipy.sparse import csr_matrix
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
    
def clean_training_data(X, y):
 
  regr = linear_model.LinearRegression()

  # Train the model using the training sets
  regr.fit(X, y)
  y_pred = regr.predict(X)
  #print("before:")
  #print(np.mean((y - y_pred)**2)) 
  for i in range(499,-1,-1): #This removes observations that 
  #are strongly affecting the entire model fit
    resi = (y_pred[i]-y[i])**2
    if(resi>900):
      X=np.delete(X,i,0)
      y=np.delete(y,i,0)
  
  #print("after:")
  regr2 = linear_model.LinearRegression()

  # Train the model using the training sets
  regr2.fit(X, y)
  y_pred2 = regr.predict(X)
  
  #print(np.mean((y - y_pred2)**2)) 
  
  
  #print("SVR rbf MSE :")
  
  svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
  svr_rbf.fit(X,y)
  pre=svr_rbf.predict(X)
 # print(np.mean((y - pre)**2))
  sX = csr_matrix(X) #sparse matrix format
  
  return sX , y
  
def transform(x):
  """
  Apply a transformation to a feature vector for a single instance
  :param x: a feature vector for a single instance
  :return: a modified feature vector
  """
  # Attemped use of kernel estimator 
  #rbf_feature = RBFSampler(gamma=0.1, random_state=1)
  #x = rbf_feature.fit_transform(x)
  #if(x[13]>3000):x[13]=x[13]//3000 #scale extreme data
  #without hurting much for its effect on model
  
  poly = PolynomialFeatures(2)
  x=poly.fit_transform(x) #2nd degree polynomial interpolation
  #Kernel methods extend this idea and can induce very high
  #(even infinite) dimensional feature spaces.
  x = Normalizer().fit_transform(x) #normalize the features
  imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
  imp.fit(x)
  x=imp.transform(x) #fill in expected values
  scaler = RobustScaler()
  scaler.fit(x)
  #x=scaler.transform(x)
  return x

# ============== MODELS ============== #




def train(X, y):
  """
  Train a model
  :parma X: n x p design matrix
  :param y: response vector of length n
  :return weights: weight vector of length p
  """
  
  """
  n, p = X.shape
  
  weights = np.zeros(p)
  regr3 = linear_model.LinearRegression(fit_intercept=False)

  # Train the model using the training sets
  regr3.fit(X, y)
  
  weights = regr3.coef_ 
  print (weights)
  
  svr_rbf = SVR(kernel='linear', C=1e3)
  svr_rbf.fit(X,y)
  weights= svr_rbf.coef_.T
  print(svr_rbf.support_vectors_.shape)
  """
  #attempted use of SVR 
  svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
  svr_rbf.fit(X,y)
  pre=svr_rbf.predict(X)
  #weights =np.linalg.lstsq(X,pre)[0]
  #weights=pre
  #print(weights)
  
  #weights= np.linalg.solve(X,pre) #X inverse times b
  
  #clf = linear_model.SGDRegressor()
  #clf.fit(X, y)
  #weights=clf.coef_
  
  
  
  regr3 = linear_model.LinearRegression(fit_intercept=False)
  # Train the Linear model using the training sets 
  regr3.fit(X, y)
  print("cross validation score:")
  scores = cross_val_score(regr3, X, y, cv=10)
  print(scores)
  
  weights = regr3.coef_ 
  return weights
  
def predict(X, weights):
  
  y = np.dot(X, weights)
  return y

def MSE(y, predictions):

  mse = np.mean((y - predictions)**2)
  return mse

def train_and_predict(X_train, y_train, X_new):

  # clean the training data
  X_train_clean, y_train_clean = clean_training_data(X_train, y_train)
  # transform the training data
  X_train_transformed = np.vstack([transform(x) for x in X_train_clean])
  # transform the new data
  X_new_transformed = np.vstack([transform(x) for x in X_new])
  # learn a model
  weights = train(X_train_transformed, y_train_clean)
  # make predictions on the training data
  predictions_train = predict(X_train_transformed, weights)
  # make predictions on the new data
  predictions = predict(X_new_transformed, weights)
  # report the MSE on the training data
  train_MSE = MSE(y_train_clean, predictions_train)
  print("MSE on training data = %0.4f" % train_MSE)
  # return the predictions on the new data
  return predictions

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
    mse = metrics.mean_squared_error(y_test, y_pred) 
    var_score = metrics.r2_score(y_test, y_pred) 

    # format numbers '{0:.5f}'.format
    mse = floatFormatter(mse)
    var_score = floatFormatter(var_score)

    return (y_pred, coef, mse, var_score)

  #==============tree=================#  
    
class dt(object):
    def __init__(self,leaf_size=4):
        self.leaf_size= leaf_size
        
    def build_tree(self,dataX,dataY):
        if dataX.shape[0] <= self.leaf_size: 
            return np.array([-1,np.mean(dataY),np.nan,np.nan])
        if np.all(dataY[:]==dataY[0]): 
            return np.array([-1,dataY[0],np.nan,np.nan])
        if np.all(dataX[:]==dataX[0]): 
            return np.array([-1,dataX[0],np.nan,np.nan])
        """
        Calc the index of the best feature.
        """
        index = 0;
        cor_list = [];
        for i in range(len(dataX[0])):
            cor_list.append(abs(np.corrcoef(dataX[:,i], dataY)[0,1]))
        index = cor_list.index(max(cor_list))
        
        """
        End Calc the index of the best feature.
        """
        leaf= np.array([-1, Counter(dataY).most_common(1)[0][0], np.nan, np.nan])
        split_value=np.medium(dataX[:,index])
        left=dataX[:,index]<=split_value
        right=dataX[:,index]>split_value
        ldataX=dataX[left,:]
        ldataY=dataY[left]
        rdataX=dataX[right,:]
        rdataY=dataY[right]
        if(len(rdataY)==0 or len(ldataX)==0 ): return leaf

        ltree=self.build_tree(ldataX,ldataY)
        rtree=self.build_tree(rdataX,rdataY)
        if ltree.ndim==1:
            root=np.array([index,split_value,1,2])
        else:
            root=np.array([index,split_value,1,ltree.shape[0]+1])
        tree=np.vstack((root,ltree,rtree))
                                                
        return tree
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        res=[]
        start=int(self.tree[0,0])
        tree_height=self.tree.shape[0]
        for point in points:
            index=start
            i=0
            while(i<tree_height):
                index=self.tree[i,0]
                if index==-1:
                    break
                else:
                    index=int(index)
                if point[index] <= self.tree[i,1]:
                    i = i + 1
                    
                else:
                    i = i + int(self.tree[i,3])
                
            if index==-1:
                res.append(self.tree[i,1])
            else:
                res.append(np.nan)
        return np.array(res)

        
#sk decision tree        
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

#===========sparse Matrix==================#


