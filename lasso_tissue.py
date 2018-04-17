### Author: Edward Huang

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso

### In this script, we first train, for each drug, a Lasso CV on GDSC, and get
### and alpha for each drug. Then, for each tissue type in TCGA, we train a
### Lasso for each drug using the learned alpha on GDSC. If the tissue type
### also exists in GDSC, then train only on the corresponding samples, and then
### test on TCGA. If the tissue type does not exist in GDSC, then train on the
### entire gene x sample matrix. With the fitted Lasso model, we then predicct
### the IC50 values on the TCGA tissue type. We then reconstruct the drug x
### sample matrix as per other methods.

# CSV files names.
train_gene_expression = './data/gdsc_tcga/gdsc_expr_postCB.csv'
test_gene_expression = './data/gdsc_tcga/tcga_expr_postCB.csv'
train_drug_response = './data/gdsc_tcga/dr_matrices/gdsc_dr.csv'
test_drug_response = './data/gdsc_tcga/dr_matrices/tcga_dr.csv'

def get_drug_lasso_alpha(train_expr, train_resp):
    '''
    Trains a lasso CV for each drug, and returns dictionary.
    Key: drug name -> str
    Value: alpha -> float
    '''
    drug_alpha_dct = {}

    X_train = train_expr.values.T
    # Train a lasso CV for each drug.
    for drug in train_resp.index.values:
        y_train_tmp = train_resp.loc[drug].values
        not_nan_ind = ~np.isnan(y_train_tmp)
        y_train_tmp = y_train_tmp[not_nan_ind]
        X_train_tmp = X_train[not_nan_ind,:]
        
        reg_model = LassoCV(n_jobs=-1)
        
        reg_model.fit(X_train_tmp, y_train_tmp)
        
        drug_alpha_dct[drug] = reg_model.alpha_
        
    return drug_alpha_dct

def main():    
    # Read CSV files.
    train_expr = pd.read_csv(train_gene_expression, index_col=0)
    test_expr = pd.read_csv(test_gene_expression, index_col=0)
    train_resp = pd.read_csv(train_drug_response, index_col=0)
    test_resp = pd.read_csv(test_drug_response, index_col=0)
    
    # drug_alpha_dct = get_drug_lasso_alpha(train_expr, train_resp)
    # print drug_alpha_dct
    # TODO. Hardcoding the results to save time.
    drug_alpha_dct = {'cisplatin': 0.15164939512562223, 'cetuximab': 0.11650346122366138, 'dabrafenib': 0.16484667529888813, 'vinblastine': 0.16227033846353733, 'sunitinib': 0.4094568663908103, 'methotrexate': 0.167896919234208, 'imatinib': 0.1943338626254849, 'temozolomide': 0.06900206868485555, 'erlotinib': 0.18394707408370034, 'veliparib': 0.09784822862992298, 'trametinib': 0.12615081380543927, 'vinorelbine': 0.21735864673541477, 'bicalutamide': 0.07809977631254474, 'tamoxifen': 0.10562113184990418, 'bleomycin': 0.24965260172450165, 'gemcitabine': 0.36300365844507365, 'gefitinib': 0.10940714085053287, 'etoposide': 0.22530320773118231, 'docetaxel': 0.1523081749856849, 'lapatinib': 0.4409559098620374, 'sorafenib': 0.23658076528620609, 'doxorubicin': 0.22250448948496374, 'paclitaxel': 0.418313649273787}
    
    # Get binary matrices indicating sample-tissue membership.
    gdsc_tissue = pd.read_csv('./data/gdsc_tissue_by_sample_2.csv', index_col=0)
    tcga_tissue = pd.read_csv('./data/tcga_tissue_by_sample2.csv', index_col=0)

    # This is the drug response dictionary. Keys are (drug, sample) pairs.
    predicted_dr_dct = {}
    
    # Track tissues in both datasets.
    common_tissues = tcga_tissue.index.intersection(gdsc_tissue.index).values

    # Loop through the TCGA tissue types. We predict on all of them.
    for tissue in tcga_tissue.index.values:
        print tissue
        # If common tissues, train on GDSC samples in tissue.
        if tissue in common_tissues:
            gdsc_tissue_samples = gdsc_tissue.loc[tissue]
            gdsc_tissue_samples = gdsc_tissue_samples.iloc[gdsc_tissue_samples.nonzero()].index.values
            train_expr_tissue = train_expr[gdsc_tissue_samples]
            train_resp_tissue = train_resp[gdsc_tissue_samples]
        # Otherwise, train on all GDSC samples.
        else:
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
    dr_matrix.to_csv('./data/gdsc_tcga/dr_matrices/GDSC_TCGA_lasso_tissue_idea_1.csv')

if __name__ == '__main__':
    main()