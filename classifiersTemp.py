# -*- coding: utf-8 -*-
"""
Created on Thu May 11 22:19:53 2017

@author: limjing7
"""

#classifiers / regressors
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

def create_classifiers():
    
    ridge = Ridge(copy_X=True)
    ridge_parameters = {'alpha':[i for i in range(185, 195, 1)]}
    
    lasso = Lasso(copy_X=True)
    lasso_parameters = {'alpha':[i/20 for i in range(3,15)]}
    
    elastic = ElasticNet(l1_ratio=0.15)
    elastic_parameters = {'alpha':[i/10 for i in range(8, 15)]}
    
    lars = Lars()
    
    rbf = SVR()
    rbf_parameters = {'C':[i/2+0.5 for i in range(4)]}
    
    lin_svm = SVR(kernel='linear')
    lin_svm_parameters = {'C':[i/2+0.5 for i in range(2, 4)]}
    
    rand_for = RandomForestRegressor(min_samples_split=0.05, max_depth=5)
    
    classifiers = {}
    classifiers['ridge'] = [ridge, ridge_parameters]
    classifiers['lasso'] = [lasso, lasso_parameters]
    classifiers['elastic'] = [elastic, elastic_parameters]
    classifiers['lars'] = [lars, None]
    classifiers['rbf'] = [rbf, rbf_parameters]
    classifiers['lin_svm'] = [lin_svm, lin_svm_parameters]
    classifiers['rand_for'] = [rand_for, None]

    return classifiers