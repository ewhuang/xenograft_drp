### Author: Edward Huang

import numpy as np
import pandas as pd
import pickle

### After training lasso models after idea 2.2, this script
### gets the list of genes with non-zero weights for every drug-tissue
### pair.

test_gene_expression = './data/gdsc_tcga/tcga_expr_postCB.csv'
test_drug_response = './data/gdsc_tcga/dr_matrices/tcga_dr.csv'

def read_pickle(fname):
    '''
    Unpickles the file, and reads in the weight list.
    '''
    with open(fname, 'rb') as fp:
        weight_lst = pickle.load(fp)
    return np.nonzero(weight_lst)[0]

def main():
    test_expr = pd.read_csv(test_gene_expression, index_col=0)
    test_resp = pd.read_csv(test_drug_response, index_col=0)
    gene_lst = test_expr.index.values
    
    tcga_tissue = pd.read_csv('./data/tcga_tissue_by_sample2.csv', index_col=0)

    out = open('./data/gdsc_tcga/idea_2_lasso_weights/nonzero_gene_lists.txt', 'w')
    for tissue in tcga_tissue.index:
        for drug in test_resp.index:
            fname = './data/gdsc_tcga/idea_2_lasso_weights/%s_%s' % (tissue, drug)
            non_zero_idx_lst = read_pickle(fname)
            non_zero_genes = [gene_lst[i] for i in non_zero_idx_lst]
            # Write out the gene sets.
            out.write('%s, %s\n' % (tissue, drug))
            out.write(','.join(non_zero_genes) + '\n')
    out.close()
    
if __name__ == '__main__':
    main()