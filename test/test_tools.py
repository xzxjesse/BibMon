#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 23:38:16 2020

@author: afranio
"""

import bibmon
import pandas as pd
import pytest
import numpy as np
from datetime import datetime
from bibmon import comparative_table
from sklearn.metrics import r2_score, mean_absolute_error

def test_complete_analysis():
    
    # load data
    data = bibmon.load_real_data()
    data = data.apply(pd.to_numeric, errors='coerce')
    
    # preprocessing pipeline
    
    preproc_tr = ['remove_empty_variables',
                  'ffill_nan',
                  'remove_frozen_variables',
                  'normalize']
    
    preproc_ts = ['ffill_nan','normalize']
    
    # define training set
        
    (X_train, X_validation, 
     X_test, Y_train, 
     Y_validation, Y_test) = bibmon.train_val_test_split(data, 
                                            start_train = '2017-12-24T12:00', 
                                            end_train = '2018-01-01T00:00', 
                                           end_validation = '2018-01-02T00:00', 
                                            end_test = '2018-01-04T00:00',
                                            tags_Y = 'tag100')
                                                         
    # define the model
                                                         
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    
    model = bibmon.sklearnRegressor(reg)                                                          

    # define regression metrics
                                                         
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_error
    
    mtr = [r2_score, mean_absolute_error]
                           
    # complete analysis!
                              
    bibmon.complete_analysis(model, X_train, X_validation, X_test, 
                            Y_train, Y_validation, Y_test,
                            f_pp_train = preproc_tr,
                            f_pp_test = preproc_ts,
                            metrics = mtr, 
                            count_window_size = 3, count_limit = 2,
                            fault_start = '2018-01-02 06:00:00',
                            fault_end = '2018-01-02 09:00:00') 
    
    model.plot_importances()                                                                             

# Fixtures for test data
@pytest.fixture
def sample_data():
    """Generate synthetic data for training, validation and testing."""
    X_train = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    Y_train = pd.Series(np.random.randn(100))
    X_validation = pd.DataFrame({
        'feature1': np.random.randn(50),
        'feature2': np.random.randn(50)
    })
    Y_validation = pd.Series(np.random.randn(50))
    X_test = pd.DataFrame({
        'feature1': np.random.randn(30),
        'feature2': np.random.randn(30)
    })
    Y_test = pd.Series(np.random.randn(30))
    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test

@pytest.fixture
def model_with_y():
    """Mock model with Y variable (regression)."""
    class MockModel:
        def __init__(self):
            self.has_Y = True
            self.name = "Model with Y"
            self.lim_conf = 0.99
            self.Y_train_orig = None
            self.X_train_orig = None
            self.Y_train_pred_orig = None
            self.X_train_pred_orig = None
            self.train_time = 0.0
            self.test_time = 0.0
            self.Y_test_orig = None
            self.Y_test_pred_orig = None
            self.X_test_orig = None
            self.X_test_pred_orig = None
            self.alarms = {}
        def predict(self, X, Y=None, *args, **kwargs):
            pred = pd.Series(np.random.randn(len(X)), index=X.index)
            if Y is not None:
                self.Y_test_orig = Y
                self.Y_test_pred_orig = pred
            self.X_test_orig = X
            self.X_test_pred_orig = pred
            self.test_time = 0.1
            return pred
        def fit(self, X_train, Y_train, f_pp=None, a_pp=None, f_pp_test=None, a_pp_test=None, lim_conf=0.99, redefine_limit=False):
            self.lim_conf = lim_conf
            self.Y_train_orig = Y_train
            self.X_train_orig = X_train
            self.Y_train_pred_orig = pd.Series(np.random.randn(len(Y_train)), index=Y_train.index)
            self.X_train_pred_orig = pd.Series(np.random.randn(len(X_train)), index=X_train.index)
            self.train_time = 0.1
            return self
    return MockModel()

@pytest.fixture
def model_without_y():
    """Mock model without Y variable (reconstruction)."""
    class MockModel:
        def __init__(self):
            self.has_Y = False
            self.name = "Model without Y"
            self.lim_conf = 0.99
            self.X_train_orig = None
            self.X_train_pred_orig = None
            self.train_time = 0.0
            self.test_time = 0.0
            self.X_test_orig = None
            self.X_test_pred_orig = None
            self.alarms = {}
        def predict(self, X, Y=None, *args, **kwargs):
            pred = pd.DataFrame(np.random.randn(*X.shape), index=X.index, columns=X.columns)
            self.X_test_orig = X
            self.X_test_pred_orig = pred
            self.test_time = 0.1
            return pred
        def fit(self, X_train, Y_train, f_pp=None, a_pp=None, f_pp_test=None, a_pp_test=None, lim_conf=0.99, redefine_limit=False):
            self.lim_conf = lim_conf
            self.X_train_orig = X_train
            self.X_train_pred_orig = pd.DataFrame(np.random.randn(*X_train.shape), index=X_train.index, columns=X_train.columns)
            self.train_time = 0.1
            return self
    return MockModel()
