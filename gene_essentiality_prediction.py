### Author: Edward Huang

import argparse
from csv import reader
import numpy as np
import os
import pandas as pd
from scipy.stats import *
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split, ShuffleSplit, cross_val_score, KFold
from sklearn.linear_model import RidgeClassifier, Ridge, ElasticNet, Lasso, Lars, LogisticRegression
from sklearn.svm import SVR, SVC

#Xenograft gao, biobank
#GDSC, TCGA
#classification: F1,precision,recall
#regression: spearman correlation, c_index
#net = np.zeros((ngene,ngene)),net[i,j]=1 #i\tj 

### Script for predicting k-fold CV on each of three datasets: xenograft2,
### GDSC, and TCGA, on gene essentiality scores.

def generate_directories():
    for folder in ('./results', './results/prediction_scores'):
        if not os.path.exists(folder):
            os.makedirs(folder)

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def read_hgnc_mappings():
    '''
    Returns a dictionary mapping HGNC symbols to sets of ENSG IDs.
    '''
    hgnc_to_ensg_dct = {}
    f = open('./data/hgnc_to_ensg.txt','r')
    f.readline() # Skip the header line.
    for line in f:
        line = line.split()
        if len(line) != 2:
            continue
        ensg_id, hgnc_symbol = line
        # Update the mapping dictionary.
        if hgnc_symbol not in hgnc_to_ensg_dct:
            hgnc_to_ensg_dct[hgnc_symbol] = set([])
        hgnc_to_ensg_dct[hgnc_symbol].add(ensg_id)
    f.close()
    return hgnc_to_ensg_dct

def read_gdsc_drug_response():
    '''
    Maps each drug to the cell lines that have response values for that drug.
    The drug response dictionary creates a dataframe of a drug x sample matrix.
    '''
    def read_drug_id_to_name_dct():
        '''
        Maps drug IDs to their names for the GDSC data.
        '''
        drug_id_to_name_dct = {}
        f = open('./data/GDSC/Screened_Compounds.txt', 'r')
        f.readline() # Skip the header line.asss
        for line in f:
            line = line.strip().split('\t')
            assert len(line) == 5
            drug_id, drug_name = line[:2]
            assert drug_id not in drug_id_to_name_dct
            drug_id_to_name_dct[drug_id] = drug_name.lower()
        f.close()
        return drug_id_to_name_dct

    drug_id_to_name_dct = read_drug_id_to_name_dct()

    gdsc_dr_dct = {}
    f = open('./data/GDSC/v17_fitted_dose_response.txt', 'r')
    f.readline() # Skip the header line.
    for line in f:
        line = line.strip().split('\t')
        sample_id, drug_id, ln_ic50 = line[2], line[3], line[5]
        assert sample_id.isdigit()
        # Skip unmappable drugs.
        if drug_id not in drug_id_to_name_dct:
            continue
        drug_name = drug_id_to_name_dct[drug_id]

        # Update the drug response dictionary.
        key = ('%s_%s' % (drug_name, drug_id), sample_id)
        assert key not in gdsc_dr_dct
        gdsc_dr_dct[key] = float(ln_ic50)
    f.close()
    # Now, convert the drug response dictionary to a dataframe.
    gdsc_dr_df = pd.Series(gdsc_dr_dct).unstack()
    return gdsc_dr_df

def read_tcga_drug_response():
    '''
    Reads the TCGA drug response.
    '''
    def read_tcga_spreadsheet(tcga_id_to_fname_dct, fname):
        '''
        Returns a dictionary mapping the sample filenames to their TCGA sample IDs.
        Key: sample filename -> str, e.g., 010caf19-9f62-4488-a2b5-eacdb795d66e.FPKM.txt
        Value: sample ID -> str, e.g., TCGA-OR-A5KT-01
        '''
        f = open(fname, 'r')
        it = reader(f)
        it.next()
        for sample_fname, sample_type, sample_id, race, cancer in it:
            # Only use samples that are primary tumors.
            if sample_type != 'Primary Tumor':
                continue
            # Exclude the number suffix ('-01') in the sample IDs.
            sample_id = '-'.join(sample_id.split('-')[:-1])
            assert sample_id not in tcga_id_to_fname_dct
            tcga_id_to_fname_dct[sample_id] = sample_fname
        f.close()

    # First, get mappings from filenames to sample IDs.
    tcga_id_to_fname_dct = {}
    for subfolder in listdir_fullpath('./data/TCGA/RNAseq'):
        for fname in listdir_fullpath(subfolder):
            if 'Pheno_All' in fname:
                read_tcga_spreadsheet(tcga_id_to_fname_dct, fname)

    # Read the drug response file.
    tcga_dr_dct = {}
    f = open('./data/TCGA/drug_response.txt', 'r')
    f.readline() # Skip the header line.
    for line in f:
        line = line.strip().split('\t')
        sample_id, drug, response = line[1], line[2].lower(), line[4]
        # Skip samples that are not primary tumors.
        if sample_id not in tcga_id_to_fname_dct:
            continue
        sample_fname = tcga_id_to_fname_dct[sample_id]
        assert '.txt' in sample_fname

        # Update the drug response dictionary.
        tcga_dr_dct[(drug, sample_fname)] = response
    f.close()
    # Convert drug response dictionary to dataframe.
    tcga_dr_df = pd.Series(tcga_dr_dct).unstack()
    return tcga_dr_df

def read_xeno_drug_response():
    '''
    Maps each drug to the cell lines that have response values for that drug.
    '''
    xeno_dr_dct = {}
    f = open('./data/Xenograft2/DrugResponsesAUCModels.txt', 'r')
    f.readline() # Skip the header line.
    for line in f:
        line = line.strip().split('\t')
        sample, drug, ic50 = line[0], line[2].lower(), np.log(float(line[4]))
        assert '.txt' not in sample and not sample.isdigit()

        # Update the drug response dictionary.
        key = (drug, sample)
        assert key not in xeno_dr_dct
        xeno_dr_dct[key] = ic50
    f.close()
    # Now, convert drug response dictionary to a dataframe.
    xeno_dr_df = pd.Series(xeno_dr_dct).unstack()
    return xeno_dr_df

def read_tcga_gene_expr_file(fname, gene_expr_table):
    table = pd.read_csv(fname, index_col=0)
    # Get the samples for this cancer's gene expression.
    ret = pd.concat([gene_expr_table, table], axis=1)
    assert list(table.index) == list(ret.index)
    return ret

def get_tcga_ge_table():
    tcga_table = pd.DataFrame()
    for subfolder in listdir_fullpath('./data/TCGA/RNAseq'):
        for fname in listdir_fullpath(subfolder):
            if 'FPKM' in fname:
                tcga_table = read_tcga_gene_expr_file(fname, tcga_table)
    # Log 2 transform just the TCGA matrix.
    tcga_table = tcga_table.add(0.1) # Add pseudo-counts to avoid NaN errors.
    tcga_table = np.log2(tcga_table)
    return tcga_table

def read_xeno_gene_expr():
    '''
    Returns a dataframe.
    '''
    # Read the gene expression file.
    xeno_expr_matrix, gene_list = [], []
    f = open('./data/Xenograft2/ExpressionModels.txt', 'r')
    for i, line in enumerate(f):
        line = line.split()
        if i == 0: # Process header line.
            sample_list = line[1:]
            continue
        hgnc_id, expr_lst = line[0], line[1:]
        if hgnc_id not in hgnc_to_ensg_dct:
            continue
        ensg_id_set = hgnc_to_ensg_dct[hgnc_id]
        # HGNC ids might amap to multiple ENSG ids.
        for ensg_id in ensg_id_set:
            if ensg_id in gene_list:
                continue
            # Update the gene list.
            gene_list += [ensg_id]
            xeno_expr_matrix += [expr_lst]
    f.close()

    # Convert the expression matrix to a dataframe.
    xeno_table = pd.DataFrame(data=np.array(xeno_expr_matrix), index=gene_list,
        columns=sample_list)
    return xeno_table

def read_gene_network(gene_df, gene_list, fname):
    '''
    Reads the gene-to-gene network, and converts it into a numpy matrix, where
    an entry from i to j means that there was an entry i\tj in the network.
    '''
    f = open(fname, 'r')
    for line in f:
        gene_i, gene_j = line.split()
        # Skip genes that do not have valid ENSG mappings.
        if gene_i not in hgnc_to_ensg_dct or gene_j not in hgnc_to_ensg_dct:
            continue
        gene_i_set = hgnc_to_ensg_dct[gene_i]
        gene_j_set = hgnc_to_ensg_dct[gene_j]
        # Loop through every possible pair of genes in one direction.
        for gene_i in gene_i_set:
            # Skip genes not in the gene expression matrix.
            if gene_i not in gene_list:
                continue
            # Get index of gene in the gene list.
            # gene_i_idx = gene_list.index(gene_i)
            for gene_j in gene_j_set:
                if gene_j not in gene_list:
                    continue
                # gene_j_idx = gene_list.index(gene_j)
                # gene_df[gene_i_idx, gene_j_idx] = 1
                gene_df.set_value(gene_i, gene_j, 1)
    f.close()

def compute_essential(c2g,net):
    ncell, ngene = c2g.shape
    score = np.zeros([ncell, ngene])
    for g in range(ngene):
        ngh = np.where(net[g,:]!=0)
        if len(ngh) == 1 and len(ngh[0]) == 0:
            continue
        for c in range(ncell):
            score[c,g] = np.mean(c2g[c,ngh])
        
    ess = np.nan_to_num(stats.zscore(score, axis=0)) + np.nan_to_num(stats.zscore(c2g, axis=0))
    return ess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_source', help='Type of data',
        required=True, choices=['gdsc', 'tcga', 'xeno'])
    parser.add_argument('-n', '--num_rounds', help='Number of rounds to train',
        required=True)
    parser.add_argument('-m', '--method', help='Method type.', choices=['baseline', 'essentiality'],
        required=True)    
    parser.add_argument('-f', '--num_folds', help='Number of folds.', choices=['3', '5'],
        required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    generate_directories()

    global hgnc_to_ensg_dct
    hgnc_to_ensg_dct = read_hgnc_mappings()

    # Gene expression table must be sample x gene DataFrame.
    # Drug response table must be a drug x sample DataFrame.
    if args.data_source == 'gdsc':
        ge_table = pd.read_table('./data/GDSC/sanger1018_brainarray_ensemblgene_rma.txt',
            index_col=0, header=0)
        dr_table = read_gdsc_drug_response()
    elif args.data_source == 'tcga':
        ge_table = get_tcga_ge_table()
        dr_table = read_tcga_drug_response()
    elif args.data_source == 'xeno':
        ge_table = read_xeno_gene_expr()
        dr_table = read_xeno_drug_response()

    # print ge_table.shape, dr_table.shape

    # Get the samples in both gene expression and drug response.
    ge_gene_list, ge_sample_list = list(ge_table.index.values), list(ge_table)
    dr_drug_list, dr_sample_list = dr_table.index.values, list(dr_table)
    intersect_samples = list(set(ge_sample_list).intersection(dr_sample_list))
    ge_table = ge_table[intersect_samples].transpose().as_matrix()
    dr_table = dr_table[intersect_samples].as_matrix()

    if args.method == 'essentiality':
        # num_genes = len(ge_gene_list)
        # gene_net = np.zeros([num_genes, num_genes])
        gene_df = pd.DataFrame([], index=ge_gene_list, columns=ge_gene_list)
        # Populate gene-gene network with zeros.
        gene_df.fillna(value=0, inplace=True)
        # Read each CRISPR network.
        for fname in listdir_fullpath('./data/CRISPR_networks'):
            read_gene_network(gene_df, ge_gene_list, fname)
        ge_table = compute_essential(ge_table, gene_df.as_matrix())

    # Predict on drug response. GDSC needs regression.
    # This spearmanr only returns the correlation, not the p-value.
    if args.data_source in ['gdsc', 'xeno']:
        personal_spearmanr = lambda x, y: stats.spearmanr(x, y)[0]
        spearman_scorer = make_scorer(personal_spearmanr)
        score_type_dct = {'spearman':spearman_scorer}
        class_dct = {'ridge':Ridge(), 'lasso':Lasso(),'elastic':ElasticNet(
            l1_ratio=0.15), 'lars':Lars(), 'rand_for_r':RandomForestRegressor(
            max_depth=5, min_samples_split=0.05), 'linear_svm':SVR(
            kernel='linear'), 'rbf_svr':SVR()}
    elif args.data_source == 'tcga':
        score_type_dct = {'precision':make_scorer(precision_score), 'recall':make_scorer(
            recall_score), 'f1':make_scorer(f1_score)}
        class_dct = {'ridge':RidgeClassifier(), 'rand_for_c':RandomForestClassifier(
            max_depth=5, min_samples_split=0.05), 'linear_svc':SVC(
            kernel='linear'), 'rbf_svc':SVC(), 'logit':LogisticRegression()}

    score_dct = {}
    # TODO: these are the 12 drugs across all three datasets.
    all_drug_lst = ['bicalutamide', 'cisplatin', 'docetaxel', 'erlotinib', 'gefitinib', 'lapatinib', 'paclitaxel', 'sorafenib', 'tamoxifen', 'temozolomide', 'vinblastine']

    for classifier_name in class_dct:
        # Get the classifier object.
        clf = class_dct[classifier_name]
        for drug_idx, drug_row in enumerate(dr_table):
            if dr_drug_list[drug_idx].split('_')[0] not in all_drug_lst:
                continue
            print dr_drug_list[drug_idx]
            # Get the non-NaN indices.
            if args.data_source in ['gdsc', 'xeno']:
                nan_idx_lst = [i for i, e in enumerate(drug_row) if not np.isnan(e)]
            else:
                nan_idx_lst = [i for i, e in enumerate(drug_row) if e != None]

            drug_row, drug_ge_table = drug_row[nan_idx_lst], ge_table[nan_idx_lst]

            # Skip drugs with very few drug responses.
            if len(drug_row) < 5:
                continue

            # X_train, X_test, y_train, y_test = train_test_split(drug_ge_table, drug_row, test_size=0.2, random_state=0)
            # clf.fit(X_train, y_train)

            # y_pred = clf.predict(X_test)

            # Get the actual scores of CV.
            for score_name in score_type_dct:
                # Repeat the classifier for num_rounds.
                for i in range(int(args.num_rounds)):
                    # Kfold
                    key = classifier_name + '_' + score_name
                    if key not in score_dct:
                        score_dct[key] = np.array([])
                    # TODO: stratified kfold for classifier?
                # score_dct[key] = np.append(score_dct[key], stats.spearmanr(y_pred, y_test)[0])
                    score_dct[key] = np.append(score_dct[key], cross_val_score(clf,
                        drug_ge_table, drug_row, cv=KFold(n=len(drug_row),
                            n_folds=int(args.num_folds), shuffle=True),
                            scoring=score_type_dct[score_name], n_jobs=1))
                print score_dct[key]

    # Write results out to file.
    out = open('./results/prediction_scores/%s_%s_%s_%s.txt' % (args.data_source,
        args.num_rounds, args.method, args.num_folds), 'w')
    for key in score_dct:
        out.write('%s\t%f\n' % (key, np.mean(score_dct[key])))
    out.close()

if __name__ == '__main__':
    main()