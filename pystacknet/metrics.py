# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 18:33:58 2018

@author: Marios Michailidis

metrics and method to check metrics used within StackNet

"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score , mean_squared_log_error  #regression metrics
from sklearn.metrics import roc_auc_score, log_loss ,accuracy_score, f1_score ,matthews_corrcoef
import numpy as np

valid_regression_metrics=["rmse","mae","rmsle","r2","mape","smape"]
valid_classification_metrics=["auc","logloss","accuracy","f1","matthews"]

############ classification metrics ############

def auc(y_true, y_pred, sample_weight=None):    
    return roc_auc_score(y_true, y_pred, sample_weight=sample_weight)

def logloss(y_true, y_pred, sample_weight=None):    
    return log_loss(y_true, y_pred, sample_weight=sample_weight)

def accuracy(y_true, y_pred, sample_weight=None):    
    return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

def f1(y_true, y_pred, sample_weight=None):    
    return f1_score(y_true, y_pred, sample_weight=sample_weight)

def matthews(y_true, y_pred, sample_weight=None):    
    return matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)

############ regression metrics ############

def rmse(y_true, y_pred, sample_weight=None):    
    return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))

def mae(y_true, y_pred, sample_weight=None):    
    return mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)

def rmsle (y_true, y_pred, sample_weight=None):    
    return np.sqrt(mean_squared_log_error(y_true, y_pred, sample_weight=sample_weight))

def r2(y_true, y_pred, sample_weight=None):    
    return r2_score(y_true, y_pred, sample_weight=sample_weight)


def mape(y_true, y_pred, sample_weight=None):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    if sample_weight is not None:
        sample_weight = sample_weight.ravel()
    eps = 1E-15
    ape = np.abs((y_true - y_pred) / (y_true + eps)) * 100
    ape[y_true == 0] = 0
    return np.average(ape, weights=sample_weight)


def smape(y_true, y_pred, sample_weight=None):
    
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    if sample_weight is not None:
        sample_weight = sample_weight.ravel()
    eps = 1E-15
    sape = (np.abs(y_true - y_pred) / (0.5 * (np.abs(y_true) + np.abs(y_pred)) + eps)) * 100
    sape[(y_true == 0) & (y_pred == 0)] = 0
    return np.average(sape, weights=sample_weight)         


"""
metric: string or class that returns a metric given (y_true, y_pred, sample_weight=None)
Curently supported metrics are "rmse","mae","rmsle","r2","mape","smape"
"""


def check_regression_metric(metric):
    
    if type(metric) is type(None):
        raise Exception ("metric cannot be None")
    if isinstance(metric, str)  :
        if metric not in valid_regression_metrics:
            raise Exception ("The regression metric has to be one of %s " % (", ".join([str(k) for k in valid_regression_metrics])))
        if metric=="rmse":
            return rmse,metric
        elif metric=="mae":
            return mae,metric
        elif metric=="rmsle":
            return rmsle,metric       
        elif metric=="r2":
            return r2,metric      
        elif metric=="mape":
            return mape,metric      
        elif metric=="smape":
            return smape,metric    
        else :
            raise Exception ("The metric %s is not recognised " % (metric) ) 
    else : #customer metrics is given
        try:
            y_true_temp=[[1],[2],[3]]
            y_pred_temp=[[2],[1],[3]]
            y_true_temp=np.array(y_true_temp)
            y_pred_temp=np.array(y_pred_temp)            
            sample_weight_temp=[1,0.5,1]
            metric(y_true_temp,y_pred_temp,  sample_weight=sample_weight_temp )
            return metric,"custom"
            
        except:
            raise Exception ("The custom metric has to implement metric(y_true, y_pred, sample_weight=None)" ) 
            
            
"""
metric: string or class that returns a metric given (y_true, y_pred, sample_weight=None)
Curently supported metrics are "rmse","mae","rmsle","r2","mape","smape"
"""


def check_classification_metric(metric):
    
    if type(metric) is type(None):
        raise Exception ("metric cannot be None")
    if isinstance(metric, str)  :
        if metric not in valid_classification_metrics:
            raise Exception ("The classification metric has to be one of %s " % (", ".join([str(k) for k in valid_classification_metrics])))
        if metric=="auc":
            return auc,metric
        elif metric=="logloss":
            return logloss,metric
        elif metric=="accuracy":
            return accuracy,metric       
        elif metric=="r2":
            return r2,metric      
        elif metric=="f1":
            return f1,metric      
        elif metric=="matthews":
            return matthews,metric    
        else :
            raise Exception ("The metric %s is not recognised " % (metric) ) 
    else : #customer metrics is given
        try:
            y_true_temp=[[1],[0],[1]]
            y_pred_temp=[[0.4],[1],[0.2]]
            y_true_temp=np.array(y_true_temp)
            y_pred_temp=np.array(y_pred_temp)
            sample_weight_temp=[1,0.5,1]
            metric(y_true_temp,y_pred_temp,  sample_weight=sample_weight_temp )
            return metric,"custom"
            
        except:
            raise Exception ("The custom metric has to implement metric(y_true, y_pred, sample_weight=None)" ) 
            
                        
            
            
            