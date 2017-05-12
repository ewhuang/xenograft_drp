# -*- coding: utf-8 -*-
"""
Created on Thu May 11 22:19:53 2017

@author: limjing7
"""

import json

#classifiers / regressors
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

def fill_in(items):
    i2 = items.copy()
    for item in i2:
        try:
            i2[item]['args']
        except KeyError:
            i2[item]['args'] = {}
        try:
            i2[item]['cv']
        except KeyError:
            i2[item]['cv'] = {}
        for param, v in i2[item]['cv'].items():
            try:
                v['divider']
            except KeyError:
                v['divider'] = 1
            try:
                v['offset']
            except KeyError:
                v['offset'] = 0
            try:
                v['start']
            except:
                v['start'] = 0
            try:
                v['step']
            except:
                v['step'] = 1
    return items

def get_params(cv_dict):
    params = {}
    for param, v in cv_dict.items():
        params[param] = [i/v['divider']+v['offset']
                            for i in range(v['start'], v['stop'], v['step'])]
    if params == {}:
        return None
    return params

def create_classifiers(filename):
    
    with open(filename, 'r') as f:
        items = json.load(f)
    
    items = fill_in(items)
    classifiers = {}
    for item in items:
        if item == 'ridge':
            classifier = Ridge(**items[item]['args'])
            params = get_params(items[item]['cv'])
            classifiers['ridge'] = [classifier, params]
        elif item == 'lasso':
            classifier = Lasso(**items[item]['args'])
            params = get_params(items[item]['cv'])
            classifiers['lasso'] = [classifier, params]
        elif item == 'elastic':
            classifier = ElasticNet(**items[item]['args'])
            params = get_params(items[item]['cv'])
            classifiers['elastic'] = [classifier, params]
        elif item == 'lars':
            classifier = Lars(**items[item]['args'])
            params = get_params(items[item]['cv'])
            classifiers['lars'] = [classifier, params]
        elif item == 'rbf':
            classifier = SVR(**items[item]['args'])
            params = get_params(items[item]['cv'])
            classifiers['rbf'] = [classifier, params]
        elif item == 'lin_svm':
            classifier = SVR(**items[item]['args'])
            params = get_params(items[item]['cv'])
            classifiers['lin_svm'] = [classifier, params]
        elif item == 'rand_for':
            classifier = RandomForestRegressor(**items[item]['args'])
            params = get_params(items[item]['cv'])
            classifiers['rand_for'] = [classifier, params]
        else:
            print(item + " not implemented yet")
    
    return classifiers