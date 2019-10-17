# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 23:56:58 2018

@author: mimar


This module will implement StackNet[https://github.com/kaz-Anova/StackNet] , allowing for both Regression and classification. 


"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,hstack,vstack ,csc_matrix
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.base import clone
from pystacknet.metrics import check_regression_metric, check_classification_metric
from sklearn.model_selection import KFold
from sklearn.utils import check_X_y,check_array,check_consistent_length, column_or_1d
import inspect
from joblib import delayed,Parallel
import operator
import time
from sklearn.preprocessing import LabelEncoder

proba_metrics=["auc","logloss"]
non_proba_metrics=["accuracy","f1","matthews"]

#(estimator, safe=True)



####### methods for paralellism ############

def _parallel_build_estimators(estimator, X, y, sample_weight, index):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    

    if  not type(sample_weight) is type (None):
            if "sample_weight" in inspect.getfullargspec(estimator.fit).args:
                estimator.fit(X, y, sample_weight=sample_weight)
            else :
                 estimator.fit(X)
    else:
        estimator.fit(X, y)

    return estimator,index


def _parallel_predict_proba(estimator, X, index):
    
    """Private function used to compute (proba-)predictions within a job."""
    
    if   hasattr(estimator, 'predict_proba') :
         predictions = estimator.predict_proba(X)
    elif hasattr(estimator, 'predict') :
         predictions = estimator.predict(X)   
    elif hasattr(estimator, 'transform') :
         predictions = estimator.transform(X)   
    else :
        raise Exception ("Each model/algorithm needs to implement at least one of ('predict()','predict_proba()' or 'transform()' ")
        
    if  hasattr(estimator, 'predict_proba') and len(predictions.shape)==2 and predictions.shape[1]==2:
        predictions=predictions[:,1]
    elif len(predictions.shape)==2 and predictions.shape[1]==1:
         predictions=predictions[:,0]
                  

    return predictions,index

def _parallel_predict_proba_scoring(estimators, X, index):
    preds=None
    """Private function used to compute (proba-)predictions within a job."""
    for estimator in estimators:
        if   hasattr(estimator, 'predict_proba') :
             predictions = estimator.predict_proba(X)
        elif hasattr(estimator, 'predict') :
             predictions = estimator.predict(X)   
        elif hasattr(estimator, 'transform') :
             predictions = estimator.transform(X)   
        else :
            raise Exception ("Each model/algorithm needs to implement at least one of ('predict()','predict_proba()' or 'transform()' ")
            
        if  hasattr(estimator, 'predict_proba') and len(predictions.shape)==2 and predictions.shape[1]==2:
            predictions=predictions[:,1]
        elif len(predictions.shape)==2 and predictions.shape[1]==1:
             predictions=predictions[:,0]
             
        if type(preds) is type(None):
            preds=predictions
        else :
            if predictions.shape!=preds.shape:
                
                raise Exception (" predictions' shape not equal among estimators within the  batch as %d!=%d " % (predictions.shape[1],preds.shape[1]))
                
            preds+=predictions
    preds/=float(len(estimators))

    return preds,index


def _parallel_predict(estimator, X, index):
    
    """Private function used to compute (proba-)predictions within a job."""
    
    if hasattr(estimator, 'predict') :
         predictions = estimator.predict(X)   
    elif hasattr(estimator, 'transform') :
         predictions = estimator.transform(X)   
    else :
        raise Exception ("Each model/algorithm needs to implement at least one of ('predict()' or 'transform()' ")
        
                  
    return predictions,index


def predict_from_broba(probas):
    preds=np.zeros(probas.shape[0]) 
    
    if len(probas.shape)==1:
        preds[probas>=0.5]=1.
    else :
        preds=np.argmax(probas, axis=1) 
    return preds
        



########################### Classifier #########################

"""

models: List of models. This should be a 2-dimensional list . The first level hould defice the stacking level and each entry is the model.  
metric: Can be "auc","logloss","accuracy","f1","matthews" or your own custom metric as long as it implements (ytrue,ypred,sample_weight=)
folds: This can be either integer to define the number of folds used in StackNet or an iterable yielding train/test splits. 
restacking: True for restacking (https://github.com/kaz-Anova/StackNet#restacking-mode) else False
use_proba : When evaluating the metric, it will use probabilities instead of class predictions if use_proba==True
use_retraining : If True it does one model based on the whole training data in order to score the test data. Otherwise it takes the average of all models used in the folds ( however this takes more memory and there is no guarantee that it will work better.) 
random_state :  Integer for randomised procedures
n_jobs :  Number of models to run in parallel. This is independent of any extra threads allocated from the selected algorithms. e.g. it is possible to run 4 models in parallel where one is a randomforest that runs on 10 threads (it selected).
verbose	: Integer value higher than zero to allow printing at the console. 

"""


class StackNetClassifier(BaseEstimator, ClassifierMixin):
    
  def __init__(self, models, metric="logloss", folds=3, restacking=False, use_retraining=True, use_proba=True, random_state=12345, n_jobs=1, verbose=0):
    
    #check models 
    if type(models) is type(None):
        raise Exception("Models cannot be None. It needs to be a list of sklearn type of models ")
    if not isinstance(models, list):     
         raise Exception("Models has to be a list of sklearn type of models ")
    for l in range (len(models)):
        if not isinstance(models[l], list):     
                 raise Exception("Each element in the models' list has to be a list . In other words a 2-dimensional list is epected. ")         
        for m in range (len(models[l])):
            if not hasattr(models[l][m], 'fit') :
                raise Exception("Each model/algorithm needs to implement a 'fit() method ")                         
            
            if not hasattr(models[l][m], 'predict_proba') and not hasattr(models[l][m], 'predict') and not hasattr(models[l][m], 'transform') :
                raise Exception("Each model/algorithm needs to implement at least one of ('predict()','predict_proba()' or 'transform()' ")                         
    self.models= models
    
    #check metrics
    self.metric,self.metric_name=check_classification_metric(metric)  
    
    #check kfold
    if not isinstance(folds, int):  
             try:
                 object_iterator = iter(folds)
             except TypeError as te:
                 raise Exception( 'folds is not int nor iterable')
    else:
        if folds <2:
             raise Exception( 'folds must be 2 or more')
             
    self.folds=folds            
    #check use_proba
    if use_proba not in [True, False]:
         raise Exception("use_proba has to be True or False")
         
    if self.metric_name in  non_proba_metrics and   use_proba==True:
        self.use_proba=False
    else :
        self.use_proba=use_proba
    
    self.layer_legths=[]

    #check restacking
    
    if restacking not in [True, False]:
         raise Exception("restacking has to be True (to include previous inputs/layers to current layers in stacking) or False")
    self.restacking= restacking

    #check retraining
    
    if use_retraining not in [True, False]:
         raise Exception("use_retraining has to be True or False")
         
    self.use_retraining= use_retraining
    
    #check random state    
    if not isinstance(random_state, int):
         raise Exception("random_state has to be int")    
    self.random_state= random_state

    #check verbose 
    if not isinstance(verbose, int):
         raise Exception("Cerbose has to be int") 

    #check verbose 
    self.n_jobs= n_jobs
    if self.n_jobs<=0:
        self.n_jobs=-1
    
    if not isinstance(n_jobs, int):
         raise Exception("n_jobs has to be int")       
         
    self.verbose= verbose
    
    
    
    self.n_classes_=None
    self.classes_ = None
    self.n_features_=None
    self.estimators_=None
    self._n_samples=None
    self._sparse=None
    self._label_encoder=None
    self._level_dims=None
    
    
  def fit (self, X, y, sample_weight=None):
        
        start_time = time.time()


        # Convert data (X is required to be 2d and indexable)
        X, y = check_X_y(
            X, y, ['csr', 'csc'], dtype=None, force_all_finite=False,
            multi_output=True
        )
        
        if  isinstance(X, list):
            X=np.array(X)        
        
        if isinstance(X, csr_matrix) or isinstance(X, csc_matrix):
            self._sparse=True
        else :
             self._sparse=False
             if len(X.shape)==1:
                X=X.reshape((X.shape[0],1))
                
        
        if type(sample_weight) is not type(None):
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(y, sample_weight)

        # Remap output
        self._n_samples, self.n_features_ = X.shape
        self._validate_y(y)
        
        self._label_encoder=LabelEncoder()
        y=self._label_encoder.fit_transform(y)
        
        
        classes = np.unique(y)
        #print (classes)
        if len(classes)<=1:
            raise Exception ("Number of classes must be at least 2, here only %d was given " %(len(classes)))
            
        self.classes_=classes
        self.n_classes_=len(self.classes_)
        
        if  isinstance(self.folds, int)   :
             indices=KFold( n_splits=self.folds,shuffle=True, random_state=self.random_state).split(y) 
             
        else :
            indices=self.folds
            
        self._level_dims =[]  
             
        previous_input=None #holds previous data for restackng 
        current_input=X
        
        self.estimators_=[]
        ##start the level training 
        for level in range (len(self.models)):
            start_level_time = time.time()
            
            if self.verbose>0:
                print ("====================== Start of Level %d ======================" % (level))            
            
            if not type(previous_input) is type(None) and self.restacking:
                if self._sparse:
                    
                    current_input=csr_matrix(hstack( [csr_matrix(previous_input), csr_matrix(current_input)]  ))
                else :
                    
                    current_input=np.column_stack((previous_input,current_input)  )
                    
            if self.verbose>0:
                print ("Input Dimensionality %d at Level %d " % (current_input.shape[1], level)) 
                
            this_level_models=self.models[level] 
            
            if self.verbose>0:
                print ("%d models included in Level %d " % (len(this_level_models), level)) 
            
            
            train_oof=None
            metrics=[0.0 for k in range(len(this_level_models))]
            
            
            indices=[t for t in indices]

            iter_count=len(indices)
            #print ("iter_count",iter_count)
            
            i=0
            #print (i)
            #print (indices)
            for train_index, test_index in indices:
                
                #print ( i, i, i)
                metrics_i=[0.0 for k in range(len(this_level_models))]
                
                X_train, X_cv = current_input[train_index], current_input[test_index]
                y_train, y_cv = y[train_index], y[test_index]
                w_train,w_cv=None,None
                if not type(sample_weight) is type (None):
                    w_train, w_cv = sample_weight[train_index], sample_weight[test_index]
                
        
                all_results = Parallel(n_jobs=min(self.n_jobs,len(this_level_models)), verbose=0)(
                    delayed(_parallel_build_estimators)(
                        clone(this_level_models[d]),
                        X_train,
                        y_train,
                        w_train, d)
                    for d in range(len(this_level_models)))
        
                # Reduce
                this_level_estimators_ = [ [t[0],t[1]] for t in all_results]
                
                this_level_estimators_=sorted(this_level_estimators_, key=operator.itemgetter(1), reverse=False)
                
                if self.use_retraining==False:
                    fitted_estimators=[t[0] for t in  this_level_estimators_]
                    if i==0:
                        self.estimators_.append([fitted_estimators]) #add level
                    else :
                        self.estimators_[level].append(fitted_estimators)
                
                #parallel predict
                all_results = Parallel(n_jobs=min(self.n_jobs,len(this_level_models)), verbose=0)(
                    delayed(_parallel_predict_proba)(
                        this_level_estimators_[d][0],
                        X_cv,d)                
                    for d in range(len(this_level_models)))
                this_level_predictions_ = [ [t[0],t[1]] for t in all_results]
                
                this_level_predictions_=sorted(this_level_predictions_, key=operator.itemgetter(1), reverse=False) 
                predictions_=[t[0] for t in  this_level_predictions_]
                
                for d in range (len(this_level_models)):
                    this_model=this_level_models[d]
                    if  self.use_proba:
                        if hasattr(this_model, 'predict_proba') :
                            metrics_i[d]=self.metric(y_cv,predictions_[d], sample_weight=w_cv)
                            metrics[d]+=metrics_i[d]
                            if self.verbose>0:
                                print ("Fold %d/%d , model %d , %s===%f " % (i+1, iter_count, d, self.metric_name, metrics_i[d]))
                        elif self.n_classes_==2 and hasattr(this_model, 'predict'):
                            metrics_i[d]=self.metric(y_cv,predictions_[d], sample_weight=w_cv)
                            metrics[d]+=metrics_i[d]
                            if self.verbose>0:
                                print ("Fold %d/%d , model %d , %s===%f " % (i+1, iter_count, d, self.metric_name, metrics_i[d]))
                            
                    
                    else :
                        if hasattr(this_model, 'predict_proba') :
                            preds_transformed=predict_from_broba(predictions_[d])
                            metrics_i[d]=self.metric(y_cv,preds_transformed, sample_weight=w_cv) #
                            metrics[d]+=metrics_i[d]
                            if self.verbose>0:
                                print ("Level %d, fold %d/%d , model %d , %s===%f " % (level, i+1, iter_count, d, self.metric_name, metrics_i[d]))
                        elif self.n_classes_==2 and hasattr(this_model, 'predict'):                
                            preds_transformed=predict_from_broba(predictions_[d])
                            metrics_i[d]=self.metric(y_cv,preds_transformed, sample_weight=w_cv) #
                            metrics[d]+=metrics_i[d]
                            if self.verbose>0:
                                print ("Level %d, fold %d/%d , model %d , %s===%f " % (level, i+1, iter_count, d, self.metric_name, metrics_i[d]))
                
                
                #concatenate predictions  
                preds_concat_=np.column_stack( predictions_)
                #print ("preds_concat_.shape", preds_concat_.shape)
                if type(train_oof) is type(None):
                    train_oof=np.zeros ( (current_input.shape[0], preds_concat_.shape[1]))
                    self._level_dims.append(preds_concat_.shape[1])

                
                if self._level_dims[level]!=preds_concat_.shape[1]:
                    raise Exception ("Output dimensionality among folds is not consistent as %d!=%d " % ( self._level_dims[level],preds_concat_.shape[1]))
                train_oof[test_index] = preds_concat_
                if self.verbose>0:
                    print ("=========== end of fold %i in level %d ===========" %(i+1,level))
                i+=1
                
            metrics=np.array(metrics)
            metrics/=float(iter_count)
            
            if self.verbose>0:
                for d in range(len(this_level_models)):
                    this_model=this_level_models[d]
                    if hasattr(this_model, 'predict_proba') :
                         print ("Level %d, model %d , %s===%f " % (level, d, self.metric_name, metrics[d]))
                    
                    
            #done cv
                        
            if self.use_retraining:
                
                all_results = Parallel(n_jobs=min(self.n_jobs,len(this_level_models)), verbose=0)(
                    delayed(_parallel_build_estimators)(
                        clone(this_level_models[d]),
                        current_input,
                        y,
                        sample_weight, d)
                    for d in range(len(this_level_models)))              
                
                
                this_level_estimators_ = [ [t[0],t[1]] for t in all_results]
                
                this_level_estimators_=sorted(this_level_estimators_, key=operator.itemgetter(1), reverse=False)

                fitted_estimators=[t[0] for t in  this_level_estimators_]

                self.estimators_.append([fitted_estimators]) #add level   
            
                
            previous_input=current_input
            current_input=train_oof
            if self.verbose>0:
                print ("Output dimensionality of level %d is %d " % ( level,current_input.shape[1] ))             
            
            
           
            end_of_level_time=time.time()
            if self.verbose>0:            
                print ("====================== End of Level %d ======================" % (level))  
                print (" level %d lasted %f seconds " % (level,end_of_level_time-start_level_time ))
        
        end_of_fit_time=time.time()        
        if self.verbose>0:          
            
            print ("====================== End of fit ======================")  
            print (" fit() lasted %f seconds " % (end_of_fit_time-start_time )) 
            
            
  # fit method that returns all out of fold predictions/outputs for all levels
  #each ith entry is a stack of oof predictions for the ith level
          
  def fit_oof (self, X, y, sample_weight=None):
        
        start_time = time.time()


        # Convert data (X is required to be 2d and indexable)
        X, y = check_X_y(
            X, y, ['csr', 'csc'], dtype=None, force_all_finite=False,
            multi_output=True
        )
        
        if  isinstance(X, list):
            X=np.array(X)        
        
        if isinstance(X, csr_matrix) or isinstance(X, csc_matrix):
            self._sparse=True
        else :
             self._sparse=False
             if len(X.shape)==1:
                X=X.reshape((X.shape[0],1))
                
        
        if type(sample_weight) is not type(None):
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(y, sample_weight)

        # Remap output
        self._n_samples, self.n_features_ = X.shape
        self._validate_y(y)
        
        self._label_encoder=LabelEncoder()
        y=self._label_encoder.fit_transform(y)
        
        out_puts=[]
        
        classes = np.unique(y)
        #print (classes)
        if len(classes)<=1:
            raise Exception ("Number of classes must be at least 2, here only %d was given " %(len(classes)))
            
        self.classes_=classes
        self.n_classes_=len(self.classes_)
        
        if  isinstance(self.folds, int)   :
             indices=KFold( n_splits=self.folds,shuffle=True, random_state=self.random_state).split(y) 
             
        else :
            indices=self.folds
            
        self._level_dims =[]  
             
        previous_input=None #holds previous data for restackng 
        current_input=X
        
        self.estimators_=[]
        ##start the level training 
        for level in range (len(self.models)):
            start_level_time = time.time()
            
            if self.verbose>0:
                print ("====================== Start of Level %d ======================" % (level))            
            
            if not type(previous_input) is type(None) and self.restacking:
                if self._sparse:
                    
                    current_input=csr_matrix(hstack( [csr_matrix(previous_input), csr_matrix(current_input)]  ))
                else :
                    
                    current_input=np.column_stack((previous_input,current_input)  )
                    
            if self.verbose>0:
                print ("Input Dimensionality %d at Level %d " % (current_input.shape[1], level)) 
                
            this_level_models=self.models[level] 
            
            if self.verbose>0:
                print ("%d models included in Level %d " % (len(this_level_models), level)) 
            
            
            train_oof=None
            metrics=[0.0 for k in range(len(this_level_models))]
            
            
            indices=[t for t in indices]

            iter_count=len(indices)
            #print ("iter_count",iter_count)
            
            i=0
            #print (i)
            #print (indices)
            for train_index, test_index in indices:
                
                #print ( i, i, i)
                metrics_i=[0.0 for k in range(len(this_level_models))]
                
                X_train, X_cv = current_input[train_index], current_input[test_index]
                y_train, y_cv = y[train_index], y[test_index]
                w_train,w_cv=None,None
                if not type(sample_weight) is type (None):
                    w_train, w_cv = sample_weight[train_index], sample_weight[test_index]
                
        
                all_results = Parallel(n_jobs=min(self.n_jobs,len(this_level_models)), verbose=0)(
                    delayed(_parallel_build_estimators)(
                        clone(this_level_models[d]),
                        X_train,
                        y_train,
                        w_train, d)
                    for d in range(len(this_level_models)))
        
                # Reduce
                this_level_estimators_ = [ [t[0],t[1]] for t in all_results]
                
                this_level_estimators_=sorted(this_level_estimators_, key=operator.itemgetter(1), reverse=False)
                
                if self.use_retraining==False:
                    fitted_estimators=[t[0] for t in  this_level_estimators_]
                    if i==0:
                        self.estimators_.append([fitted_estimators]) #add level
                    else :
                        self.estimators_[level].append(fitted_estimators)
                
                #parallel predict
                all_results = Parallel(n_jobs=min(self.n_jobs,len(this_level_models)), verbose=0)(
                    delayed(_parallel_predict_proba)(
                        this_level_estimators_[d][0],
                        X_cv,d)                
                    for d in range(len(this_level_models)))
                this_level_predictions_ = [ [t[0],t[1]] for t in all_results]
                
                this_level_predictions_=sorted(this_level_predictions_, key=operator.itemgetter(1), reverse=False) 
                predictions_=[t[0] for t in  this_level_predictions_]
                
                for d in range (len(this_level_models)):
                    this_model=this_level_models[d]
                    if  self.use_proba:
                        if hasattr(this_model, 'predict_proba') :
                            metrics_i[d]=self.metric(y_cv,predictions_[d], sample_weight=w_cv)
                            metrics[d]+=metrics_i[d]
                            if self.verbose>0:
                                print ("Fold %d/%d , model %d , %s===%f " % (i+1, iter_count, d, self.metric_name, metrics_i[d]))
                        elif self.n_classes_==2 and hasattr(this_model, 'predict'):
                            metrics_i[d]=self.metric(y_cv,predictions_[d], sample_weight=w_cv)
                            metrics[d]+=metrics_i[d]
                            if self.verbose>0:
                                print ("Fold %d/%d , model %d , %s===%f " % (i+1, iter_count, d, self.metric_name, metrics_i[d]))
                            
                    
                    else :
                        if hasattr(this_model, 'predict_proba') :
                            preds_transformed=predict_from_broba(predictions_[d])
                            metrics_i[d]=self.metric(y_cv,preds_transformed, sample_weight=w_cv) #
                            metrics[d]+=metrics_i[d]
                            if self.verbose>0:
                                print ("Level %d, fold %d/%d , model %d , %s===%f " % (level, i+1, iter_count, d, self.metric_name, metrics_i[d]))
                        elif self.n_classes_==2 and hasattr(this_model, 'predict'):                
                            preds_transformed=predict_from_broba(predictions_[d])
                            metrics_i[d]=self.metric(y_cv,preds_transformed, sample_weight=w_cv) #
                            metrics[d]+=metrics_i[d]
                            if self.verbose>0:
                                print ("Level %d, fold %d/%d , model %d , %s===%f " % (level, i+1, iter_count, d, self.metric_name, metrics_i[d]))
                
                
                #concatenate predictions  
                preds_concat_=np.column_stack( predictions_)
                

                
                #print ("preds_concat_.shape", preds_concat_.shape)
                if type(train_oof) is type(None):
                    train_oof=np.zeros ( (current_input.shape[0], preds_concat_.shape[1]))
                    self._level_dims.append(preds_concat_.shape[1])

                
                if self._level_dims[level]!=preds_concat_.shape[1]:
                    raise Exception ("Output dimensionality among folds is not consistent as %d!=%d " % ( self._level_dims[level],preds_concat_.shape[1]))
                train_oof[test_index] = preds_concat_
                if self.verbose>0:
                    print ("=========== end of fold %i in level %d ===========" %(i+1,level))
                i+=1
                
            metrics=np.array(metrics)
            metrics/=float(iter_count)
            
            if self.verbose>0:
                for d in range(len(this_level_models)):
                    this_model=this_level_models[d]
                    if hasattr(this_model, 'predict_proba') :
                         print ("Level %d, model %d , %s===%f " % (level, d, self.metric_name, metrics[d]))
                    
                    
            #done cv
                        
            if self.use_retraining:
                
                all_results = Parallel(n_jobs=min(self.n_jobs,len(this_level_models)), verbose=0)(
                    delayed(_parallel_build_estimators)(
                        clone(this_level_models[d]),
                        current_input,
                        y,
                        sample_weight, d)
                    for d in range(len(this_level_models)))              
                
                
                this_level_estimators_ = [ [t[0],t[1]] for t in all_results]
                
                this_level_estimators_=sorted(this_level_estimators_, key=operator.itemgetter(1), reverse=False)

                fitted_estimators=[t[0] for t in  this_level_estimators_]

                self.estimators_.append([fitted_estimators]) #add level   
            
            out_puts.append(train_oof)  
            
            previous_input=current_input
            current_input=train_oof
            if self.verbose>0:
                print ("Output dimensionality of level %d is %d " % ( level,current_input.shape[1] ))             
            
            
           
            end_of_level_time=time.time()
            if self.verbose>0:            
                print ("====================== End of Level %d ======================" % (level))  
                print (" level %d lasted %f seconds " % (level,end_of_level_time-start_level_time ))
        
        end_of_fit_time=time.time()        
        if self.verbose>0:          
            
            print ("====================== End of fit ======================")  
            print (" fit() lasted %f seconds " % (end_of_fit_time-start_time )) 
            
        return out_puts
            
            
  def predict_proba (self, X):
        
        if type(self.n_classes_) is type(None) or self.n_classes_==1:
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")
        if type(self.classes_) is type(None) or len(self.classes_)==1:
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")   
        if type(self.n_features_) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")             
        if type(self.estimators_) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ") 
        if type(self._n_samples) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")             
        if type(self._sparse) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")             
        if type(self._label_encoder) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")             
        if type(self._level_dims) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")   
    
        if  isinstance(X, list):
            X=np.array(X)        
        
        predict_sparse=None
        if isinstance(X, csr_matrix) or isinstance(X, csc_matrix):
            predict_sparse=True
        else :
            predict_sparse=False
            if len(X.shape)==1:
                X=X.reshape((X.shape[0],1))    
                
        if X.shape[1]!=self.n_features_:
            raise Exception("Input dimensionality of %d is not the same as the trained one with %d " % ( X.shape[1], self.n_features_))


        # Remap output
        predict_sparse_samples, predict_sparse_n_features_ = X.shape        
        
        previous_input=None #holds previous data for restackng 
        current_input=X
        
        ##start the level training 
        
        for level in range (len(self.estimators_)):
            #start_level_time = time.time()
            
            if self.verbose>0:
                print ("====================== Start of Level %d ======================" % (level))            
            
            if not type(previous_input) is type(None) and self.restacking:
                if predict_sparse:
                    
                    current_input=csr_matrix(hstack( [csr_matrix(previous_input), csr_matrix(current_input)]  ))
                else :
                    
                    current_input=np.column_stack((previous_input,current_input)  )        
        
            this_level_estimators=self.estimators_[level] 
            
            if self.verbose>0:
                print ("%d estimators included in Level %d " % (len(this_level_estimators), level)) 
            
            

            all_results = Parallel(n_jobs=min(self.n_jobs,len(this_level_estimators[0])), verbose=0)(
                delayed(_parallel_predict_proba_scoring)(
                    [this_level_estimators[s][d] for s in range (len(this_level_estimators))],
                    current_input,d)                
                for d in range(len(this_level_estimators[0])))
                
            this_level_predictions_ = [ [t[0],t[1]] for t in all_results]
            
            this_level_predictions_=sorted(this_level_predictions_, key=operator.itemgetter(1), reverse=False) 
            predictions_=[t[0] for t in  this_level_predictions_]
            

            #concatenate predictions  
            test_pred=np.column_stack( predictions_)  
            if test_pred.shape[1]!= self._level_dims[level]:
                raise Exception ("Output dimensionality for level %d with %d is not the same as the one during training with %d " %(level,test_pred.shape[1], self._level_dims[level] ))
            
            previous_input=current_input
            current_input=test_pred       
        
        if len(test_pred.shape)==2 and test_pred.shape[1]==1 :
             pr=np.zeros( (test_pred.shape[0],2))
             pr[:,1]=test_pred[:,0]
             pr[:,0]=1-test_pred[:,0]
             test_pred=pr 
        elif len(test_pred.shape)==1:
             pr=np.zeros( (test_pred.shape[0],2))
             pr[:,1]=test_pred
             pr[:,0]=1-test_pred
             test_pred=pr             
        return test_pred
            
  #predicts output up to the specified level
          
  def predict_up_to(self, X, lev=None):
        
        if type(self.n_classes_) is type(None) or self.n_classes_==1:
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")
        if type(self.classes_) is type(None) or len(self.classes_)==1:
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")   
        if type(self.n_features_) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")             
        if type(self.estimators_) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ") 
        if type(self._n_samples) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")             
        if type(self._sparse) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")             
        if type(self._label_encoder) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")             
        if type(self._level_dims) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")   
    
        if  isinstance(X, list):
            X=np.array(X)        
        
        predict_sparse=None
        if isinstance(X, csr_matrix) or isinstance(X, csc_matrix):
            predict_sparse=True
        else :
            predict_sparse=False
            if len(X.shape)==1:
                X=X.reshape((X.shape[0],1))    
                
        if X.shape[1]!=self.n_features_:
            raise Exception("Input dimensionality of %d is not the same as the trained one with %d " % ( X.shape[1], self.n_features_))


        # Remap output
        predict_sparse_samples, predict_sparse_n_features_ = X.shape        
        
        previous_input=None #holds previous data for restackng 
        current_input=X
        
        if type(lev) is type(None):
            lev=len(self.estimators_)
        
        if not isinstance(lev, int):
            raise Exception("lev has to be int") 
         
        out_puts=[]    
        lev=min(lev,len(self.estimators_) )
        
        ##start the level training 
        
        for level in range (lev):
            #start_level_time = time.time()
            
            if self.verbose>0:
                print ("====================== Start of Level %d ======================" % (level))            
            
            if not type(previous_input) is type(None) and self.restacking:
                if predict_sparse:
                    
                    current_input=csr_matrix(hstack( [csr_matrix(previous_input), csr_matrix(current_input)]  ))
                else :
                    
                    current_input=np.column_stack((previous_input,current_input)  )        
        
            this_level_estimators=self.estimators_[level] 
            
            if self.verbose>0:
                print ("%d estimators included in Level %d " % (len(this_level_estimators), level)) 
            
            

            all_results = Parallel(n_jobs=min(self.n_jobs,len(this_level_estimators[0])), verbose=0)(
                delayed(_parallel_predict_proba_scoring)(
                    [this_level_estimators[s][d] for s in range (len(this_level_estimators))],
                    current_input,d)                
                for d in range(len(this_level_estimators[0])))
                
            this_level_predictions_ = [ [t[0],t[1]] for t in all_results]
            
            this_level_predictions_=sorted(this_level_predictions_, key=operator.itemgetter(1), reverse=False) 
            predictions_=[t[0] for t in  this_level_predictions_]
            

            #concatenate predictions  
            test_pred=np.column_stack( predictions_)  
            if test_pred.shape[1]!= self._level_dims[level]:
                raise Exception ("Output dimensionality for level %d with %d is not the same as the one during training with %d " %(level,test_pred.shape[1], self._level_dims[level] ))
            
            out_puts.append(test_pred)
            
            previous_input=current_input
            current_input=test_pred       
        
           
        return out_puts
                        
            
        
        
  def _validate_y(self, y):
        if len(y.shape) == 1 or y.shape[1] == 1:
            return column_or_1d(y, warn=True)
        else:
            return y        
        





########################### Regression #########################


        
"""

models: List of models. This should be a 2-dimensional list . The first level hould defice the stacking level and each entry is the model.  
metric: Can be "rmse","mae","rmsle","r2","mape","smape" or your own custom metric as long as it implements (ytrue,ypred,sample_weight=)
folds: This can be either integer to define the number of folds used in StackNet or an iterable yielding train/test splits. 
restacking: True for restacking (https://github.com/kaz-Anova/StackNet#restacking-mode) else False
use_retraining : If True it does one model based on the whole training data in order to score the test data. Otherwise it takes the average of all models used in the folds ( however this takes more memory and there is no guarantee that it will work better.) 
random_state :  Integer for randomised procedures
n_jobs :  Number of models to run in parallel. This is independent of any extra threads allocated from the selected algorithms. e.g. it is possible to run 4 models in parallel where one is a randomforest that runs on 10 threads (it selected).
verbose	: Integer value higher than zero to allow printing at the console. 

"""


class StackNetRegressor(BaseEstimator, RegressorMixin):
    
  def __init__(self, models, metric="rmse", folds=3, restacking=False, use_retraining=True, random_state=12345, n_jobs=1, verbose=0):
    
    #check models 
    if type(models) is type(None):
        raise Exception("Models cannot be None. It needs to be a list of sklearn type of models ")
    if not isinstance(models, list):     
         raise Exception("Models has to be a list of sklearn type of models ")
    for l in range (len(models)):
        if not isinstance(models[l], list):     
                 raise Exception("Each element in the models' list has to be a list . In other words a 2-dimensional list is epected. ")         
        for m in range (len(models[l])):
            if not hasattr(models[l][m], 'fit') :
                raise Exception("Each model/algorithm needs to implement a 'fit() method ")                         
            
            if not hasattr(models[l][m], 'predict_proba') and not hasattr(models[l][m], 'predict') and not hasattr(models[l][m], 'transform') :
                raise Exception("Each model/algorithm needs to implement at least one of ('predict()','predict_proba()' or 'transform()' ")                         
    self.models= models
    
    #check metrics
    self.metric,self.metric_name=check_regression_metric(metric)  
    
    #check kfold
    if not isinstance(folds, int):  
             try:
                 object_iterator = iter(folds)
             except TypeError as te:
                 raise Exception( 'folds is not int nor iterable')
    else:
        if folds <2:
             raise Exception( 'folds must be 2 or more')
             
    self.folds=folds            
    
    self.layer_legths=[]

    #check restacking
    
    if restacking not in [True, False]:
         raise Exception("restacking has to be True (to include previous inputs/layers to current layers in stacking) or False")
    self.restacking= restacking

    #check retraining
    
    if use_retraining not in [True, False]:
         raise Exception("use_retraining has to be True or False")
         
    self.use_retraining= use_retraining
    
    #check random state    
    if not isinstance(random_state, int):
         raise Exception("random_state has to be int")    
    self.random_state= random_state

    #check verbose 
    if not isinstance(verbose, int):
         raise Exception("Cerbose has to be int") 

    #check verbose 
    self.n_jobs= n_jobs
    if self.n_jobs<=0:
        self.n_jobs=-1
    
    if not isinstance(n_jobs, int):
         raise Exception("n_jobs has to be int")       
         
    self.verbose= verbose

    self.n_features_=None
    self.estimators_=None
    self._n_samples=None
    self._sparse=None
    self._level_dims=None
    
    
  def fit (self, X, y, sample_weight=None):
        start_time = time.time()


        # Convert data (X is required to be 2d and indexable)
        X, y = check_X_y(
            X, y, ['csr', 'csc'], dtype=None, force_all_finite=False,
            multi_output=True
        )
        
        if  isinstance(X, list):
            X=np.array(X)        
        
        if isinstance(X, csr_matrix) or isinstance(X, csc_matrix):
            self._sparse=True
        else :
             self._sparse=False
             if len(X.shape)==1:
                X=X.reshape((X.shape[0],1))
                
        
        if type(sample_weight) is not type(None):
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(y, sample_weight)

        # Remap output
        self._n_samples, self.n_features_ = X.shape
        self._validate_y(y)
        
        

        if  isinstance(self.folds, int)   :
             indices=KFold( n_splits=self.folds,shuffle=True, random_state=self.random_state).split(y) 
             
        else :
            indices=self.folds
            
        self._level_dims =[]  
             
        previous_input=None #holds previous data for restackng 
        current_input=X
        
        self.estimators_=[]
        ##start the level training 
        for level in range (len(self.models)):
            start_level_time = time.time()
            
            if self.verbose>0:
                print ("====================== Start of Level %d ======================" % (level))            
            
            if not type(previous_input) is type(None) and self.restacking:
                if self._sparse:
                    
                    current_input=csr_matrix(hstack( [csr_matrix(previous_input), csr_matrix(current_input)]  ))
                else :
                    
                    current_input=np.column_stack((previous_input,current_input)  )
                    
            if self.verbose>0:
                print ("Input Dimensionality %d at Level %d " % (current_input.shape[1], level)) 
                
            this_level_models=self.models[level] 
            
            if self.verbose>0:
                print ("%d models included in Level %d " % (len(this_level_models), level)) 
            
            
            train_oof=None
            metrics=[0.0 for k in range(len(this_level_models))]
            
            
            indices=[t for t in indices]

            iter_count=len(indices)
            #print ("iter_count",iter_count)
            
            i=0
            #print (i)
            #print (indices)
            for train_index, test_index in indices:
                
                #print ( i, i, i)
                metrics_i=[0.0 for k in range(len(this_level_models))]
                
                X_train, X_cv = current_input[train_index], current_input[test_index]
                y_train, y_cv = y[train_index], y[test_index]
                w_train,w_cv=None,None
                if not type(sample_weight) is type (None):
                    w_train, w_cv = sample_weight[train_index], sample_weight[test_index]
                
        
                all_results = Parallel(n_jobs=min(self.n_jobs,len(this_level_models)), verbose=0)(
                    delayed(_parallel_build_estimators)(
                        clone(this_level_models[d]),
                        X_train,
                        y_train,
                        w_train, d)
                    for d in range(len(this_level_models)))
        
                # Reduce
                this_level_estimators_ = [ [t[0],t[1]] for t in all_results]
                
                this_level_estimators_=sorted(this_level_estimators_, key=operator.itemgetter(1), reverse=False)
                
                if self.use_retraining==False:
                    fitted_estimators=[t[0] for t in  this_level_estimators_]
                    if i==0:
                        self.estimators_.append([fitted_estimators]) #add level
                    else :
                        self.estimators_[level].append(fitted_estimators)
                
                #parallel predict
                all_results = Parallel(n_jobs=min(self.n_jobs,len(this_level_models)), verbose=0)(
                    delayed(_parallel_predict_proba)(
                        this_level_estimators_[d][0],
                        X_cv,d)                
                    for d in range(len(this_level_models)))
                this_level_predictions_ = [ [t[0],t[1]] for t in all_results]
                
                this_level_predictions_=sorted(this_level_predictions_, key=operator.itemgetter(1), reverse=False) 
                predictions_=[t[0] for t in  this_level_predictions_]
                
                for d in range (len(this_level_models)):
                    this_model=this_level_models[d]
                    if hasattr(this_model, 'predict') :
                        metrics_i[d]=self.metric(y_cv,predictions_[d], sample_weight=w_cv)
                        metrics[d]+=metrics_i[d]
                        if self.verbose>0:
                            print ("Fold %d/%d , model %d , %s===%f " % (i+1, iter_count, d, self.metric_name, metrics_i[d]))
                    elif  predictions_[d].shape==y_cv.shape  :                       
                        metrics_i[d]=self.metric(y_cv,predictions_[d], sample_weight=w_cv)
                        metrics[d]+=metrics_i[d]
                        if self.verbose>0:
                            print ("Fold %d/%d , model %d , %s===%f " % (i+1, iter_count, d, self.metric_name, metrics_i[d]))

                
                #concatenate predictions  
                preds_concat_=np.column_stack( predictions_)
                #print ("preds_concat_.shape", preds_concat_.shape)
                if type(train_oof) is type(None):
                    train_oof=np.zeros ( (current_input.shape[0], preds_concat_.shape[1]))
                    self._level_dims.append(preds_concat_.shape[1])

                
                if self._level_dims[level]!=preds_concat_.shape[1]:
                    raise Exception ("Output dimensionality among folds is not consistent as %d!=%d " % ( self._level_dims[level],preds_concat_.shape[1]))
                train_oof[test_index] = preds_concat_
                if self.verbose>0:
                    print ("=========== end of fold %i in level %d ===========" %(i+1,level))
                i+=1
                
            metrics=np.array(metrics)
            metrics/=float(iter_count)
            
            if self.verbose>0:
                for d in range(len(this_level_models)):
                    this_model=this_level_models[d]
                    if hasattr(this_model, 'predict_proba') :
                         print ("Level %d, model %d , %s===%f " % (level, d, self.metric_name, metrics[d]))
                    
                    
            #done cv
                        
            if self.use_retraining:
                
                all_results = Parallel(n_jobs=min(self.n_jobs,len(this_level_models)), verbose=0)(
                    delayed(_parallel_build_estimators)(
                        clone(this_level_models[d]),
                        current_input,
                        y,
                        sample_weight, d)
                    for d in range(len(this_level_models)))              
                
                
                this_level_estimators_ = [ [t[0],t[1]] for t in all_results]
                
                this_level_estimators_=sorted(this_level_estimators_, key=operator.itemgetter(1), reverse=False)

                fitted_estimators=[t[0] for t in  this_level_estimators_]

                self.estimators_.append([fitted_estimators]) #add level   
            
                
            previous_input=current_input
            current_input=train_oof
            if self.verbose>0:
                print ("Output dimensionality of level %d is %d " % ( level,current_input.shape[1] ))             
            
            
           
            end_of_level_time=time.time()
            if self.verbose>0:            
                print ("====================== End of Level %d ======================" % (level))  
                print (" level %d lasted %f seconds " % (level,end_of_level_time-start_level_time ))
        
        end_of_fit_time=time.time()        
        if self.verbose>0:          
            
            print ("====================== End of fit ======================")  
            print (" fit() lasted %f seconds " % (end_of_fit_time-start_time )) 
            
            
  # fit method that returns all out of fold predictions/outputs for all levels
  #each ith entry is a stack of oof predictions for the ith level
          
  def fit_oof (self, X, y, sample_weight=None):
        
        start_time = time.time()


        # Convert data (X is required to be 2d and indexable)
        X, y = check_X_y(
            X, y, ['csr', 'csc'], dtype=None, force_all_finite=False,
            multi_output=True
        )
        
        if  isinstance(X, list):
            X=np.array(X)        
        
        if isinstance(X, csr_matrix) or isinstance(X, csc_matrix):
            self._sparse=True
        else :
             self._sparse=False
             if len(X.shape)==1:
                X=X.reshape((X.shape[0],1))
                
        
        if type(sample_weight) is not type(None):
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(y, sample_weight)

        # Remap output
        self._n_samples, self.n_features_ = X.shape
        self._validate_y(y)
        
        
        out_puts=[]
        
        if  isinstance(self.folds, int)   :
             indices=KFold( n_splits=self.folds,shuffle=True, random_state=self.random_state).split(y) 
             
        else :
            indices=self.folds
            
        self._level_dims =[]  
             
        previous_input=None #holds previous data for restackng 
        current_input=X
        
        self.estimators_=[]
        ##start the level training 
        for level in range (len(self.models)):
            start_level_time = time.time()
            
            if self.verbose>0:
                print ("====================== Start of Level %d ======================" % (level))            
            
            if not type(previous_input) is type(None) and self.restacking:
                if self._sparse:
                    
                    current_input=csr_matrix(hstack( [csr_matrix(previous_input), csr_matrix(current_input)]  ))
                else :
                    
                    current_input=np.column_stack((previous_input,current_input)  )
                    
            if self.verbose>0:
                print ("Input Dimensionality %d at Level %d " % (current_input.shape[1], level)) 
                
            this_level_models=self.models[level] 
            
            if self.verbose>0:
                print ("%d models included in Level %d " % (len(this_level_models), level)) 
            
            
            train_oof=None
            metrics=[0.0 for k in range(len(this_level_models))]
            
            
            indices=[t for t in indices]

            iter_count=len(indices)
            #print ("iter_count",iter_count)
            
            i=0
            #print (i)
            #print (indices)
            for train_index, test_index in indices:
                
                #print ( i, i, i)
                metrics_i=[0.0 for k in range(len(this_level_models))]
                
                X_train, X_cv = current_input[train_index], current_input[test_index]
                y_train, y_cv = y[train_index], y[test_index]
                w_train,w_cv=None,None
                if not type(sample_weight) is type (None):
                    w_train, w_cv = sample_weight[train_index], sample_weight[test_index]
                
        
                all_results = Parallel(n_jobs=min(self.n_jobs,len(this_level_models)), verbose=0)(
                    delayed(_parallel_build_estimators)(
                        clone(this_level_models[d]),
                        X_train,
                        y_train,
                        w_train, d)
                    for d in range(len(this_level_models)))
        
                # Reduce
                this_level_estimators_ = [ [t[0],t[1]] for t in all_results]
                
                this_level_estimators_=sorted(this_level_estimators_, key=operator.itemgetter(1), reverse=False)
                
                if self.use_retraining==False:
                    fitted_estimators=[t[0] for t in  this_level_estimators_]
                    if i==0:
                        self.estimators_.append([fitted_estimators]) #add level
                    else :
                        self.estimators_[level].append(fitted_estimators)
                
                #parallel predict
                all_results = Parallel(n_jobs=min(self.n_jobs,len(this_level_models)), verbose=0)(
                    delayed(_parallel_predict_proba)(
                        this_level_estimators_[d][0],
                        X_cv,d)                
                    for d in range(len(this_level_models)))
                this_level_predictions_ = [ [t[0],t[1]] for t in all_results]
                
                this_level_predictions_=sorted(this_level_predictions_, key=operator.itemgetter(1), reverse=False) 
                predictions_=[t[0] for t in  this_level_predictions_]
                
                for d in range (len(this_level_models)):
                    this_model=this_level_models[d]
                    if hasattr(this_model, 'predict') :
                        metrics_i[d]=self.metric(y_cv,predictions_[d], sample_weight=w_cv)
                        metrics[d]+=metrics_i[d]
                        if self.verbose>0:
                            print ("Fold %d/%d , model %d , %s===%f " % (i+1, iter_count, d, self.metric_name, metrics_i[d]))
                    elif  predictions_[d].shape==y_cv.shape  :                       
                        metrics_i[d]=self.metric(y_cv,predictions_[d], sample_weight=w_cv)
                        metrics[d]+=metrics_i[d]
                        if self.verbose>0:
                            print ("Fold %d/%d , model %d , %s===%f " % (i+1, iter_count, d, self.metric_name, metrics_i[d]))

                
                #concatenate predictions  
                preds_concat_=np.column_stack( predictions_)
                #print ("preds_concat_.shape", preds_concat_.shape)
                if type(train_oof) is type(None):
                    train_oof=np.zeros ( (current_input.shape[0], preds_concat_.shape[1]))
                    self._level_dims.append(preds_concat_.shape[1])

                
                if self._level_dims[level]!=preds_concat_.shape[1]:
                    raise Exception ("Output dimensionality among folds is not consistent as %d!=%d " % ( self._level_dims[level],preds_concat_.shape[1]))
                train_oof[test_index] = preds_concat_
                if self.verbose>0:
                    print ("=========== end of fold %i in level %d ===========" %(i+1,level))
                i+=1
                
            metrics=np.array(metrics)
            metrics/=float(iter_count)
            
            if self.verbose>0:
                for d in range(len(this_level_models)):
                    this_model=this_level_models[d]
                    if hasattr(this_model, 'predict_proba') :
                         print ("Level %d, model %d , %s===%f " % (level, d, self.metric_name, metrics[d]))
                    
                    
            #done cv
                        
            if self.use_retraining:
                
                all_results = Parallel(n_jobs=min(self.n_jobs,len(this_level_models)), verbose=0)(
                    delayed(_parallel_build_estimators)(
                        clone(this_level_models[d]),
                        current_input,
                        y,
                        sample_weight, d)
                    for d in range(len(this_level_models)))              
                
                
                this_level_estimators_ = [ [t[0],t[1]] for t in all_results]
                
                this_level_estimators_=sorted(this_level_estimators_, key=operator.itemgetter(1), reverse=False)

                fitted_estimators=[t[0] for t in  this_level_estimators_]

                self.estimators_.append([fitted_estimators]) #add level   
            
            out_puts.append(train_oof)  
            
            previous_input=current_input
            current_input=train_oof
            if self.verbose>0:
                print ("Output dimensionality of level %d is %d " % ( level,current_input.shape[1] ))             
            
            
           
            end_of_level_time=time.time()
            if self.verbose>0:            
                print ("====================== End of Level %d ======================" % (level))  
                print (" level %d lasted %f seconds " % (level,end_of_level_time-start_level_time ))
        
        end_of_fit_time=time.time()        
        if self.verbose>0:          
            
            print ("====================== End of fit ======================")  
            print (" fit() lasted %f seconds " % (end_of_fit_time-start_time )) 
            
        return out_puts
            
            
  def predict (self, X):
        

        if type(self.n_features_) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")             
        if type(self.estimators_) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ") 
        if type(self._n_samples) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")             
        if type(self._sparse) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")                      
        if type(self._level_dims) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")   
    
        if  isinstance(X, list):
            X=np.array(X)        
        
        predict_sparse=None
        if isinstance(X, csr_matrix) or isinstance(X, csc_matrix):
            predict_sparse=True
        else :
            predict_sparse=False
            if len(X.shape)==1:
                X=X.reshape((X.shape[0],1))    
                
        if X.shape[1]!=self.n_features_:
            raise Exception("Input dimensionality of %d is not the same as the trained one with %d " % ( X.shape[1], self.n_features_))


        # Remap output
        predict_sparse_samples, predict_sparse_n_features_ = X.shape        
        
        previous_input=None #holds previous data for restackng 
        current_input=X
        
        ##start the level training 
        
        for level in range (len(self.estimators_)):
            #start_level_time = time.time()
            
            if self.verbose>0:
                print ("====================== Start of Level %d ======================" % (level))            
            
            if not type(previous_input) is type(None) and self.restacking:
                if predict_sparse:
                    
                    current_input=csr_matrix(hstack( [csr_matrix(previous_input), csr_matrix(current_input)]  ))
                else :
                    
                    current_input=np.column_stack((previous_input,current_input)  )        
        
            this_level_estimators=self.estimators_[level] 
            
            if self.verbose>0:
                print ("%d estimators included in Level %d " % (len(this_level_estimators), level)) 
            
            

            all_results = Parallel(n_jobs=min(self.n_jobs,len(this_level_estimators[0])), verbose=0)(
                delayed(_parallel_predict_proba_scoring)(
                    [this_level_estimators[s][d] for s in range (len(this_level_estimators))],
                    current_input,d)                
                for d in range(len(this_level_estimators[0])))
                
            this_level_predictions_ = [ [t[0],t[1]] for t in all_results]
            
            this_level_predictions_=sorted(this_level_predictions_, key=operator.itemgetter(1), reverse=False) 
            predictions_=[t[0] for t in  this_level_predictions_]
            

            #concatenate predictions  
            test_pred=np.column_stack( predictions_)  
            if test_pred.shape[1]!= self._level_dims[level]:
                raise Exception ("Output dimensionality for level %d with %d is not the same as the one during training with %d " %(level,test_pred.shape[1], self._level_dims[level] ))
            
            previous_input=current_input
            current_input=test_pred       
        
           
        return test_pred
            
  #predicts output up to the specified level
          
  def predict_up_to(self, X, lev=None):
        
        if type(self.n_features_) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")             
        if type(self.estimators_) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ") 
        if type(self._n_samples) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")             
        if type(self._sparse) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")                         
        if type(self._level_dims) is type(None) :
            raise Exception ("fit() must run successfuly to be able to execute the current method. ")   
    
        if  isinstance(X, list):
            X=np.array(X)        
        
        predict_sparse=None
        if isinstance(X, csr_matrix) or isinstance(X, csc_matrix):
            predict_sparse=True
        else :
            predict_sparse=False
            if len(X.shape)==1:
                X=X.reshape((X.shape[0],1))    
                
        if X.shape[1]!=self.n_features_:
            raise Exception("Input dimensionality of %d is not the same as the trained one with %d " % ( X.shape[1], self.n_features_))


        # Remap output
        predict_sparse_samples, predict_sparse_n_features_ = X.shape        
        
        previous_input=None #holds previous data for restackng 
        current_input=X
        
        if type(lev) is type(None):
            lev=len(self.estimators_)
        
        if not isinstance(lev, int):
            raise Exception("lev has to be int") 
            
        lev=min(lev,len(self.estimators_) )
        out_puts=[]
        ##start the level training 
        for level in range (lev):
            #start_level_time = time.time()
            
            if self.verbose>0:
                print ("====================== Start of Level %d ======================" % (level))            
            
            if not type(previous_input) is type(None) and self.restacking:
                if predict_sparse:
                    
                    current_input=csr_matrix(hstack( [csr_matrix(previous_input), csr_matrix(current_input)]  ))
                else :
                    
                    current_input=np.column_stack((previous_input,current_input)  )        
        
            this_level_estimators=self.estimators_[level] 
            
            if self.verbose>0:
                print ("%d estimators included in Level %d " % (len(this_level_estimators), level)) 
            
            

            all_results = Parallel(n_jobs=min(self.n_jobs,len(this_level_estimators[0])), verbose=0)(
                delayed(_parallel_predict_proba_scoring)(
                    [this_level_estimators[s][d] for s in range (len(this_level_estimators))],
                    current_input,d)                
                for d in range(len(this_level_estimators[0])))
                
            this_level_predictions_ = [ [t[0],t[1]] for t in all_results]
            
            this_level_predictions_=sorted(this_level_predictions_, key=operator.itemgetter(1), reverse=False) 
            predictions_=[t[0] for t in  this_level_predictions_]
            

            #concatenate predictions  
            test_pred=np.column_stack( predictions_)
            print (test_pred.shape)
            if test_pred.shape[1]!= self._level_dims[level]:
                raise Exception ("Output dimensionality for level %d with %d is not the same as the one during training with %d " %(level,test_pred.shape[1], self._level_dims[level] ))
            
            out_puts.append(test_pred)
            previous_input=current_input
            current_input=test_pred       
        
            
        return out_puts
                        
            
        
        
  def _validate_y(self, y):
        if len(y.shape) == 1 or y.shape[1] == 1:
            return column_or_1d(y, warn=True)
        else:
            return y          
    
