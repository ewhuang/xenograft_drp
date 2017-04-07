# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 13:21:51 2017

@author: limjing7
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from scipy import stats


ridge = Ridge(copy_X=True)
ridge_parameters = {'alpha':[i/3 for i in range(10)]}

elastic = ElasticNet()
elastic_parameters = {'l1_ratio':[i/3 for i in range(0,10)]}

lars = Lars()

classifiers = {}
classifiers['ridge'] = [ridge, ridge_parameters]
classifiers['elastic'] = [elastic, elastic_parameters]
classifiers['lars'] = [lars, None]

def regress(rounds, classifiers):
    
    gene_exp = pd.read_csv("gdsc/permutation_"+str(0)+".csv", index_col=0)
    drug_response = pd.read_excel("gdsc/v17_fitted_dose_response.xlsx")
    d1_response = drug_response[drug_response['DRUG_ID']==1]
    #rmv drug_response data that uses cell-lines that do not have exp data
    d1_response = d1_response.loc[[str(i) in gene_exp for i in d1_response['COSMIC_ID']]]
    
    cell_lines = d1_response['COSMIC_ID']
    classifiers_name = ['ridge', 'elastic', 'lars']
    columns = [classifier+'_'+str(round) for classifier in classifiers_name for round in range(rounds)]
    
    cor_indices = ['spearman correlation', 'spearman pval', 'pearson correlation', 'pearson pval']
    correlation = pd.DataFrame(index = cor_indices, columns=columns)
    
    columns.append('true_val')
    
    result = pd.DataFrame(index = map(str, cell_lines), columns=columns)
    
    for a in d1_response['COSMIC_ID']:
        val = d1_response[d1_response['COSMIC_ID']==a]['LN_IC50'].values[0]
        result.loc[str(a), 'true_val'] = val
    
    for round in range(rounds):
        gene_exp = pd.read_csv("gdsc/permutation_"+str(round)+".csv", index_col=0)
        
        to_rmv = []
        for cell_line in gene_exp:
            try: 
                if len(d1_response[d1_response['COSMIC_ID']==int(cell_line)])==0:
                    to_rmv.append(cell_line)
            except ValueError:
                to_rmv.append(cell_line)
        mask = gene_exp.columns.isin(to_rmv)
        
        g_useful = gene_exp.loc[:,~mask]
        g_useful = g_useful.apply(stats.zscore, axis=1)
            
        
        g_wolab = gene_exp.iloc[:, 1:]
        g_t = np.transpose(g_useful)
        
        kf = KFold(n_splits=5)
        folds = kf.split(g_t)
        
        t_val_shape = result.loc[:, 'true_val'].values.shape[0]
        true_values = result.loc[:, 'true_val'].values.reshape(t_val_shape)     #for correlation
        
        for train, test in folds:
            to_rmv = []
            train_set = g_t.iloc[train, :]
            ntrain = np.transpose(train_set)
            
            n_train_y = []
            for example in ntrain:
                d_resp = d1_response[d1_response['COSMIC_ID']==int(example)].loc[:,'LN_IC50']
                n_train_y.append(d_resp.values[0])
            ntrain_t = np.transpose(ntrain)
            
            to_rmv = []
            test_set = g_t.iloc[test, :]
            ntest = np.transpose(test_set)
            
            n_test_y = []
            for example in ntest:
                d_resp = d1_response[d1_response['COSMIC_ID']==int(example)].loc[:,'LN_IC50']
                n_test_y.append(d_resp.values[0])
            ntest_t = np.transpose(ntest)
            
            for c_name, c_vals in classifiers.items():
                if (c_vals[1] is not None):
                    classifier = GridSearchCV(c_vals[0], param_grid=c_vals[1])
                    classifier.fit(ntrain_t, y=n_train_y)
                    
                    estimator = classifier.best_estimator_
                    preds = estimator.predict(ntest_t)
                    for index, cell_line in enumerate(ntest):
                        try:
                            result.loc[cell_line, c_name+'_'+str(round)] = preds[index]
                        except KeyError:
                            pass
                else:
                    c_vals[0].fit(ntrain_t, y=n_train_y)
                    preds = c_vals[0].predict(ntest_t)
                    print(ntest.columns)
                    for index, cell_line in enumerate(ntest):
                        try:
                            result.loc[cell_line, c_name+'_'+str(round)] = preds[index]
                        except KeyError:
                            pass
        for c_name in classifiers:
            c_pred_shape = result.loc[:, c_name+'_'+str(round)].shape[0]
            curr_preds = result.loc[:, c_name+'_'+str(round)].reshape(c_pred_shape)
            
            s_corr, s_pval = stats.spearmanr(curr_preds, true_values)
            p_corr, p_pval = stats.pearsonr(curr_preds, true_values)
            
            correlation.loc['spearman correlation', c_name+'_'+str(round)] = s_corr
            correlation.loc['spearman pval', c_name+'_'+str(round)] = s_pval
            correlation.loc['pearson correlation', c_name+'_'+str(round)] = p_corr
            correlation.loc['pearson pval', c_name+'_'+str(round)] = p_pval
    #            
    #        classifier2 = GridSearchCV(elastic, param_grid=elastic_parameters)
    #        classifier2.fit(ntrain_t, y=n_train_y)
    #        
    #        estimator = classifier2.best_estimator_
    #        preds = estimator.predict(ntest_t)
    #        print(ntest.columns)
    #        for index, cell_line in enumerate(ntest):
    #            try:
    #                result.loc[cell_line, 'elastic_'+str(round)] = preds[index]
    #            except KeyError:
    #                pass
    #            
    #        lars.fit(ntrain_t, y=n_train_y)
    #        preds = lars.predict(ntest_t)
    #        print(ntest.columns)
    #        for index, cell_line in enumerate(ntest):
    #            try:
    #                result.loc[cell_line, 'lars_'+str(round)] = preds[index]
    #            except KeyError:
    #                pass
    #    break
    return result, correlation
#result.to_csv('gdsc/results2.csv')

result, correlation = regress(1, classifiers)