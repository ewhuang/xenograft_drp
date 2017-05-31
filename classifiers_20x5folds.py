# -*- coding: utf-8 -*-
"""
Performs regression with different classifiers
and calculates correlation coefficients

@author: limjing7
"""

import argparse
import datetime
import numpy as np
import pandas as pd

import createClassifiers

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from scipy import stats

def regress(rounds, classifiers, dnumber=1):
    """Performs regression with 5-fold cross_validation
       
    
    Parameters
    ----------
    rounds: number of rounds of repeats done
    classifiers: dictionary of classifiers used
        - key: name to for classifier identification
        - value: list of 2 values: [classifier, dictionary of hyperparameters]
    dnumber: drug number in gdsc dataset
    
    """
    
    
    gene_exp = pd.read_csv("gdsc/permutation_"+str(0)+".csv", index_col=0)
    drugs_response = pd.read_excel("gdsc/v17_fitted_dose_response.xlsx")
    drug_response = drugs_response[drugs_response['DRUG_ID']==dnumber]
    #rmv drugs_response data that uses cell-lines that do not have exp data
    drug_response = drug_response.loc[[str(i) in gene_exp for i in drug_response['COSMIC_ID']]]
    
    cell_lines = drug_response['COSMIC_ID']
    classifiers_name = ['ridge', 'elastic', 'lars']
    columns = [classifier+'_'+str(round) for classifier in classifiers_name for round in range(rounds)]
    
    cor_indices = ['spearman correlation', 'spearman pval', 'pearson correlation', 'pearson pval']
    correlation = pd.DataFrame(index = cor_indices, columns=[])
    
    columns.append('true_val')
    
    result = pd.DataFrame(index = map(str, cell_lines), columns=[])
    best = []
    
    for a in drug_response['COSMIC_ID']:
        val = drug_response[drug_response['COSMIC_ID']==a]['LN_IC50'].values[0]
        result.loc[str(a), 'true_val'] = val
    
    for round in range(rounds):
        print("Round: "+str(round))
        gene_exp = pd.read_csv("gdsc/permutation_"+str(round)+".csv", index_col=0)
        
        to_rmv = []
        for cell_line in gene_exp:
            try: 
                if len(drug_response[drug_response['COSMIC_ID']==int(cell_line)])==0:
                    to_rmv.append(cell_line)
            except ValueError:
                to_rmv.append(cell_line)
        mask = gene_exp.columns.isin(to_rmv)
        
        g_useful = gene_exp.loc[:,~mask]
        g_useful = g_useful.apply(stats.zscore, axis=1)
            
        
        #g_wolab = gene_exp.iloc[:, 1:]
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
                d_resp = drug_response[drug_response['COSMIC_ID']==int(example)].loc[:,'LN_IC50']
                n_train_y.append(d_resp.values[0])
            ntrain_t = np.transpose(ntrain)
            
            to_rmv = []
            test_set = g_t.iloc[test, :]
            ntest = np.transpose(test_set)
            
            n_test_y = []
            for example in ntest:
                d_resp = drug_response[drug_response['COSMIC_ID']==int(example)].loc[:,'LN_IC50']
                n_test_y.append(d_resp.values[0])
            ntest_t = np.transpose(ntest)
            
            for c_name, c_vals in classifiers.items():
                if (c_vals[1] is not None):
#                    print('starting')
                    classifier = GridSearchCV(c_vals[0], param_grid=c_vals[1])
                    classifier.fit(ntrain_t, y=n_train_y)
#                    print('stopping')
                    
                    estimator = classifier.best_estimator_
                    best.append(estimator)
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
            curr_preds = result.loc[:, c_name+'_'+str(round)].values.reshape(c_pred_shape)
            
            s_corr, s_pval = stats.spearmanr(curr_preds, true_values)
            p_corr, p_pval = stats.pearsonr(curr_preds, true_values)
            
            correlation.loc['spearman correlation', c_name+'_'+str(round)] = s_corr
            correlation.loc['spearman pval', c_name+'_'+str(round)] = s_pval
            correlation.loc['pearson correlation', c_name+'_'+str(round)] = p_corr
            correlation.loc['pearson pval', c_name+'_'+str(round)] = p_pval
            
    return result, correlation, best
#result.to_csv('gdsc/results2.csv')

def reg_average(corr, classifiers, rounds):
    """Calculates the average 
    
    Parameters
    ----------
    corr : dataframe containing correlation data in columns
    classifiers: dictionary of classifiers used
    rounds: number of rounds of repeats done
    """
    
    cols = [c_name for c_name in classifiers]
    cor_indices = corr.index
    tot_corr = pd.DataFrame(index = cor_indices, columns=cols, data=0)
    for c_name in classifiers:
        for round in range(rounds):
            tot_corr[c_name] += corr[c_name+'_'+str(round)]
    tot_corr /= rounds
    return tot_corr

def main():
    parser = argparse.ArgumentParser(description='perform cv regression')
    parser.add_argument('-d', '--dnum', help='drug number to regress on',
                        default = 1, metavar='' , type = int)
    parser.add_argument('-r', '--rounds', help='number of rounds to repeat',
                        default = 20, metavar='', type = int)
    parser.add_argument('-f', '--fname', help='file containing classifiers',
                        metavar='')
    args = parser.parse_args()
    dnum = args.dnum
    rounds = args.rounds
    fname = args.fname
    print(dnum)

    classifiers = createClassifiers.create_classifiers(fname)

    start = datetime.datetime.now()
    result, corr, best = regress(rounds, classifiers, dnum)
    correlation = reg_average(corr, classifiers, 20)
#    result.to_csv('gdsc/v3/result.csv')
#    corr.to_csv('gdsc/v3/corr.csv')
#    correlation.to_csv('gdsc/v3/correlation.csv')
#    end = datetime.datetime.now()
#    
#    time_taken = str(end-start)
#    with open('gdsc/v3/time_taken.txt', 'w') as f:
#        f.write(time_taken)
#        
#    print(best)

if __name__ == "__main__":
    main()
    

