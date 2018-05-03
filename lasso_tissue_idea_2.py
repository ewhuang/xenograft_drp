### Author: Edward Huang

import argparse
import itertools
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_squared_error

### The second idea is that unlike the first idea in which you would choose one
### single alpha (hyper-parameter) for all the tissue types, here depending on
### the tissue, your alpha will be different. Assume that you want to predict
### the drug response of TCGA patients with Breast Cancer (your test set) using
### a model trained on GDSC data. To choose the alpha_breast for this task,
### during the CV using GDSC data, you place all the GDSC samples corresponding
### to the breast tissue in your validation set and choose the alpha that works
### best for that. After the alpha is selected, you can do 2 things: in Idea
### 2.1, and similar to Idea 1, you only use GDSC breast samples to train a
### model using alpha_breast and then apply it to your test set (TCGA breast
### samples) or in Idea 2.2 you use alpha_breast and train a model using all
### GDSC sample to predict drug response of your TCGA test set. 

# CSV files names.
train_gene_expression = './data/gdsc_tcga/gdsc_expr_postCB.csv'
test_gene_expression = './data/gdsc_tcga/tcga_expr_postCB.csv'
train_drug_response = './data/gdsc_tcga/dr_matrices/gdsc_dr.csv'
test_drug_response = './data/gdsc_tcga/dr_matrices/tcga_dr.csv'

# Trained from lasso_tissue.py.
ALL_TISSUE_DRUG_ALPHA_DCT = {'cisplatin': 0.15164939512562223, 'cetuximab': 0.11650346122366138, 'dabrafenib': 0.16484667529888813, 'vinblastine': 0.16227033846353733, 'sunitinib': 0.4094568663908103, 'methotrexate': 0.167896919234208, 'imatinib': 0.1943338626254849, 'temozolomide': 0.06900206868485555, 'erlotinib': 0.18394707408370034, 'veliparib': 0.09784822862992298, 'trametinib': 0.12615081380543927, 'vinorelbine': 0.21735864673541477, 'bicalutamide': 0.07809977631254474, 'tamoxifen': 0.10562113184990418, 'bleomycin': 0.24965260172450165, 'gemcitabine': 0.36300365844507365, 'gefitinib': 0.10940714085053287, 'etoposide': 0.22530320773118231, 'docetaxel': 0.1523081749856849, 'lapatinib': 0.4409559098620374, 'sorafenib': 0.23658076528620609, 'doxorubicin': 0.22250448948496374, 'paclitaxel': 0.418313649273787}

def fit_model(*args):
    X_train_tmp, y_train_tmp, X_test_tmp, y_test_tmp, alpha = args[0]
    reg_model = Lasso(alpha=alpha)
    reg_model.fit(X_train_tmp, y_train_tmp)
    
    y_pred = reg_model.predict(X_test_tmp)
    rmse = mean_squared_error(y_test_tmp, y_pred)
    return rmse

def tune_alpha(train_expr_non_tissue, train_resp_non_tissue, train_expr_tissue,
    train_resp_tissue):
    '''
    Given the training expression and training response as well as a tissue
    type, find the alpha that trains on all other tissue types and predicts on
    the given tissue type.
    '''
    drug_alpha_dct = {}
    
    X_train = train_expr_non_tissue.values.T
    X_test = train_expr_tissue.values.T
    # Get the best alpha for each drug.
    for drug in train_resp_non_tissue.index.values:
        # Get the training set.
        y_train_tmp = train_resp_non_tissue.loc[drug].values
        not_nan_ind = ~np.isnan(y_train_tmp)
        y_train_tmp = y_train_tmp[not_nan_ind]
        X_train_tmp = X_train[not_nan_ind,:]
        
        # Get the validation set.
        y_test_tmp = train_resp_tissue.loc[drug].values
        not_nan_ind = ~np.isnan(y_test_tmp)
        y_test_tmp = y_test_tmp[not_nan_ind]
        X_test_tmp = X_test[not_nan_ind,:]
        
        # If there are no non-nan samples for this tissue in GDDSC, then use the
        # alpha from training on all of GDSC.
        if X_test_tmp.shape[0] == 0:
            drug_alpha_dct[drug] = ALL_TISSUE_DRUG_ALPHA_DCT[drug]
            continue
            
        # Initialize the alpha dictionary for the current drug.
        best_alpha_dct = {}
        pool = Pool(processes=24)
        alpha_space = np.logspace(-2, -1, 100)
        rmse_lst = pool.map(fit_model, itertools.izip(itertools.repeat(X_train_tmp),
            itertools.repeat(y_train_tmp), itertools.repeat(X_test_tmp),
            itertools.repeat(y_test_tmp), alpha_space))
        for i, e in enumerate(alpha_space):
            best_alpha_dct[e] = rmse_lst[i]
        pool.close()
        pool.join()
        # Check which alpha yields the highest performance.
        # for alpha in np.logspace(-2, -1, 100):
            # reg_model = Lasso(alpha=alpha)
            # reg_model.fit(X_train_tmp, y_train_tmp)
            
            # y_pred = reg_model.predict(X_test_tmp)
            # rmse = mean_squared_error(y_test_tmp, y_pred)
            
            # best_alpha_dct[alpha] = rmse
        # Get the alpha corresponding to the lowest rmse.
        drug_alpha_dct[drug] = min(best_alpha_dct, key=best_alpha_dct.get)
    return drug_alpha_dct

def main():
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--idea', help='idea type', choices=['1', '2'])
    args = parser.parse_args()

    # Read CSV files.
    train_expr = pd.read_csv(train_gene_expression, index_col=0)
    test_expr = pd.read_csv(test_gene_expression, index_col=0)
    train_resp = pd.read_csv(train_drug_response, index_col=0)
    test_resp = pd.read_csv(test_drug_response, index_col=0)
    
    # Get binary matrices indicating sample-tissue membership.
    gdsc_tissue = pd.read_csv('./data/gdsc_tissue_by_sample_2.csv', index_col=0)
    tcga_tissue = pd.read_csv('./data/tcga_tissue_by_sample2.csv', index_col=0)

    # This is the drug response dictionary. Keys are (drug, sample) pairs.
    predicted_dr_dct = {}
    
    # Loop through the TCGA tissue types. We predict on all of them.
    for tissue in tcga_tissue.index.values:
    # for tissue in ['Brain']: # TODO
        print tissue
        # If tissue in GDSC, train on GDSC samples in tissue.
        if tissue in gdsc_tissue.index.values:
            gdsc_tissue_samples = gdsc_tissue.loc[tissue]
            gdsc_tissue_samples = gdsc_tissue_samples.iloc[gdsc_tissue_samples.nonzero()].index.values
            # Get the samples corresponding to the tissue.
            train_expr_tissue = train_expr[gdsc_tissue_samples]
            train_resp_tissue = train_resp[gdsc_tissue_samples]
            # Get all samples excluding current tissue.
            train_expr_non_tissue = train_expr.drop(gdsc_tissue_samples, axis=1)
            train_resp_non_tissue = train_resp.drop(gdsc_tissue_samples, axis=1)
            # Find the best alpha with the tissue samples as the validation set.
            drug_alpha_dct = tune_alpha(train_expr_non_tissue, train_resp_non_tissue,
                train_expr_tissue, train_resp_tissue)
        # Otherwise, train on all GDSC samples.
        else:
            # Choose alpha from the drug dictionary from the other script.
            train_expr_tissue = train_expr
            train_resp_tissue = train_resp
            
            # This dictionary is trained from lasso_tissue.py
            drug_alpha_dct = ALL_TISSUE_DRUG_ALPHA_DCT
        # For idea 2.2, we train on all GDSC samples rather than just the tissue.
        if args.idea == '2':
            train_expr_tissue = train_expr
            train_resp_tissue = train_resp

        # Do the same for TCGA.
        tcga_tissue_samples = tcga_tissue.loc[tissue]
        tcga_tissue_samples = tcga_tissue_samples.iloc[tcga_tissue_samples.nonzero()].index.values
        test_expr_tissue = test_expr[tcga_tissue_samples]
        test_resp_tissue = test_resp[tcga_tissue_samples]
        
        # Go through the drug names.
        for drug in list(train_resp.index):
            y_train_tmp = train_resp_tissue.loc[drug].values
            not_nan_ind = ~np.isnan(y_train_tmp)
            y_train_tmp = y_train_tmp[not_nan_ind]
            X_train_tmp = train_expr_tissue.values.T[not_nan_ind,:]
            
            # Remake the training set to the entire set if the tissue has no non-nan values.
            if X_train_tmp.shape[0] == 0:
                y_train_tmp = train_resp.loc[drug].values
                not_nan_ind = ~np.isnan(y_train_tmp)
                y_train_tmp = y_train_tmp[not_nan_ind]
                X_train_tmp = train_expr.values.T[not_nan_ind,:]
                
            # Use the alpha learned from training on all samples.
            clf = Lasso(alpha=drug_alpha_dct[drug])
            clf.fit(X_train_tmp, y_train_tmp)
        
            # Predict on the test values.    
            y_test_hat_tmp = clf.predict(test_expr_tissue.values.T)
            for i, ic50 in enumerate(y_test_hat_tmp):
                predicted_dr_dct[(drug, tcga_tissue_samples[i])] = ic50

    # Unstack the dictionary into a dataframe.
    dr_matrix = pd.Series(predicted_dr_dct).unstack()
    dr_matrix = dr_matrix[test_resp.columns]
    # Write out the results.
    dr_matrix.to_csv('./data/gdsc_tcga/dr_matrices/GDSC_TCGA_lasso_tissue_idea_2.%s.csv' %
        args.idea)

if __name__ == '__main__':
    main()