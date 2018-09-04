# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 23:06:17 2018

@author: mimar

uses the dataset from https://www.kaggle.com/c/amazon-employee-access-challenge/data

"""
import numpy as np
from pystacknet.pystacknet import StackNetClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn import preprocessing
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def load_data(pat, filename, use_labels=True):
    """
    Load data from CSV files and return them as numpy arrays
    The use_labels parameter indicates whether one should
    read the first column (containing class labels). If false,
    return all 0s. 
    """

    # load column 1 to 8 (ignore last one)
    data = np.loadtxt(open(pat+ filename), delimiter=',',
                      usecols=range(1, 8), skiprows=1)
    if use_labels:
        labels = np.loadtxt(open(pat + filename), delimiter=',',
                            usecols=[0], skiprows=1)
    else:
        labels = np.zeros(data.shape[0])
    return labels, data

def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))



def test_pystacknet():
    
    
    path=""
    
    

    y, X = load_data(path, 'train.csv')
    y_test, X_test = load_data(path, 'test.csv', use_labels=False)

    # === one-hot encoding === #
    # we want to encode the category IDs encountered both in
    # the training and the test set, so we fit the encoder on both
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_test)))
    X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
    X_test = encoder.transform(X_test)

    
    
    #####################################################################################
    ###############################  CLASSIFICATION #####################################        
    #####################################################################################
    
    
    models=[ 
            
            [LogisticRegression(C=1,  random_state=1),
             LogisticRegression(C=3,  random_state=1),
             Ridge(alpha=0.1, random_state=1),
             LogisticRegression(penalty="l1", C=1, random_state=1),
             XGBClassifier(max_depth=5,learning_rate=0.1, n_estimators=300, objective="binary:logistic", n_jobs=1, booster="gbtree", random_state=1, colsample_bytree=0.4 ),
             XGBClassifier(max_depth=5,learning_rate=0.3, reg_lambda=0.1, n_estimators=300, objective="binary:logistic", n_jobs=1, booster="gblinear", random_state=1, colsample_bytree=0.4 ),
             XGBClassifier(max_depth=5,learning_rate=0.1, n_estimators=300, objective="rank:pairwise", n_jobs=1, booster="gbtree", random_state=1, colsample_bytree=0.4 ),
             LGBMClassifier(boosting_type='gbdt', num_leaves=40, max_depth=-1, learning_rate=0.01, n_estimators=1000, subsample_for_bin=1000, objective="xentropy", min_split_gain=0.0, min_child_weight=0.01, min_child_samples=10, subsample=0.9, subsample_freq=1, colsample_bytree=0.5, reg_alpha=0.0, reg_lambda=0.0, random_state=1, n_jobs=1)             
             ],
            
            [RandomForestClassifier (n_estimators=300, criterion="entropy", max_depth=6, max_features=0.5, random_state=1)]
            
            
            ]
    
    
    ##################  proba metric ###############################    
    
    model=StackNetClassifier(models, metric="auc", folds=4, restacking=False,
                             use_retraining=True, use_proba=True, random_state=12345,
                             n_jobs=1, verbose=1)
    
    model.fit(X,y )
    preds=model.predict_proba(X_test)[:,1]
    
    save_results(preds,path+ "pystacknet_pred.csv")
    #print ("auc test 2 , auc %f " % (roc_auc_score(y_test,preds)))   
    

    
if __name__ == '__main__':
    test_pystacknet()