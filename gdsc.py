# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 13:00:44 2017

@author: limjing7
"""

import numpy as np
import pandas as pd

drug_response = pd.read_excel("gdsc/v17_fitted_dose_response.xlsx")
gene_exp = pd.read_table("gdsc/sanger1018_brainarray_ensemblgene_rma.txt", index_col=0)

np.random.seed(10)

for i in range(20):
    cols = gene_exp.columns.tolist()
    s_cols = cols[0:1]+list(np.random.permutation(cols[1:]))
    g_exp_shuffled = gene_exp[s_cols]
    g_exp_shuffled.to_csv('gdsc/permutation_'+str(i)+'.csv')