### Author: Edward Huang

import argparse
import csv
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import preprocess_batch_effect
from scipy.stats import *
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, Ridge, ElasticNet, Lasso, Lars, LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.svm import SVR, SVC
import warnings

### Script for predicting k-fold CV on each of three datasets: xenograft2,
### GDSC, and TCGA, on gene essentiality scores.

def generate_directories():
    for folder in ('./results', './results/prediction_scores'):
        if not os.path.exists(folder):
            os.makedirs(folder)

def read_gene_network(fname):
    '''
    Reads the gene-to-gene network, and converts it into a numpy matrix, where
    an entry from i to j means that there was an entry i\tj in the network.
    '''
    edge_dct = {}
    f = open(fname, 'r')
    for line in f:
        gene_i, gene_j = line.split()[:2]
        # # Skip genes that do not have valid ENSG mappings.
        if gene_i not in hgnc_to_ensg_dct or gene_j not in hgnc_to_ensg_dct:
            continue
        gene_i_set = hgnc_to_ensg_dct[gene_i]
        gene_j_set = hgnc_to_ensg_dct[gene_j]
        # # Loop through every possible pair of genes in one direction.
        for ensg_i in gene_i_set:
            for ensg_j in gene_j_set:
                edge_dct[(ensg_i, ensg_j)] = 1
    f.close()
    return edge_dct

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
    parser.add_argument('-m', '--method', help='Method type.', choices=[
        'baseline', 'essentiality'], required=True)
    parser.add_argument('-i', '--input_net', help='Input network file name (full path).')
    parser.add_argument('-f', '--num_folds', help='Number of folds.', required=True)
    return parser.parse_args()

def get_classifier_dct(data_source):
    '''
    Given the type of data, produce classifiers/regressions. TCGA gets classifiers.
    '''
    # This spearmanr only returns the correlation, not the p-value.
    if data_source in ['gdsc', 'xeno']:
        score_type_dct = {'spearman':lambda x, y: stats.spearmanr(x, y)[0]}
        clf_dct = {'elastic':ElasticNet(l1_ratio=0.15),
            'rand_for_r':RandomForestRegressor(max_depth=5,
                min_samples_split=0.05, n_estimators=100),
            'linear_svm':SVR(kernel='linear'), 'rbf_svr':SVR(),
            'poly_svr':SVR(kernel='poly'), 'sigmod_svr':SVR(kernel='sigmoid')}
    elif data_source == 'tcga':
        score_type_dct = {'precision':precision_score, 'recall':recall_score, 'f1':f1_score}
        clf_dct = {'ridge':RidgeClassifier(), 'rand_for_c':RandomForestClassifier(
            max_depth=5, min_samples_split=0.05, n_estimators=100),
            'linear_svc':SVC(kernel='linear'), 'poly_svc':SVC(kernel='poly'),
            'sigmod_svc':SVC(kernel='sigmoid'),'logit':LogisticRegression()}
    return score_type_dct, clf_dct

def train_classifiers(input_net, data_source, method, num_folds, xeno_type):
    # Gene expression table must be gene x sample DataFrame.
    # Drug response table must be a drug x sample DataFrame.
    if data_source == 'gdsc':
        ge_table = preprocess_batch_effect.read_gdsc_gene_expr()
        dr_table = preprocess_batch_effect.read_gdsc_drug_response()[1]
    elif data_source == 'tcga':
        ge_table = preprocess_batch_effect.read_tcga_gene_expr()
        dr_table = preprocess_batch_effect.read_tcga_drug_response()[1]
    elif data_source == 'xeno':
        ge_table = preprocess_batch_effect.read_xeno_gene_expr(xeno_type)
        dr_table = preprocess_batch_effect.read_xeno_drug_response(xeno_type)[1]

    print(ge_table.shape, dr_table.shape)

    # Get the samples in both gene expression and drug response.
    ge_gene_list, ge_sample_list = list(ge_table.index.values), list(ge_table)
    dr_drug_list, dr_sample_list = dr_table.index.values, list(dr_table)
    intersect_samples = list(set(ge_sample_list).intersection(dr_sample_list))
    ge_table = ge_table[intersect_samples].transpose().as_matrix()
    dr_table = dr_table[intersect_samples].as_matrix()

    print(ge_table.shape, dr_table.shape)

    if method == 'essentiality':
        # essentiality mode requires an input network.
        assert input_net != None and os.path.exists(input_net)
        gene_df = pd.DataFrame([], index=ge_gene_list, columns=ge_gene_list)
        edge_dct = read_gene_network(input_net)
        gene_df = gene_df.fillna(pd.Series(edge_dct).unstack())
        gene_df = gene_df.fillna(value=0)
        print('done filling missing values')
        ge_table = compute_essential(ge_table, gene_df.as_matrix())

    print('finished reading')

    score_type_dct, clf_dct = get_classifier_dct(data_source)

    score_dct = {}
    print('start training')
    for classifier_name in clf_dct:
        # Get the classifier object.
        clf = clf_dct[classifier_name]
        for drug_idx, drug_row in enumerate(dr_table):
            drug_name = dr_drug_list[drug_idx]
            # Get the non-NaN indices.
            if data_source in ['gdsc', 'xeno']:
                nan_idx_lst = [i for i, e in enumerate(drug_row) if not np.isnan(e)]
            else:
                nan_idx_lst = [i for i, e in enumerate(drug_row) if e != None]

            drug_row, drug_ge_table = drug_row[nan_idx_lst], ge_table[nan_idx_lst]
            # Skip drugs with very few drug responses.
            if len(drug_row) < num_folds or (data_source == 'tcga' and
                len(set(drug_row)) == 1):
                continue

            if data_source in ['gdsc', 'xeno']:
                cv = KFold(n_splits=int(num_folds), shuffle=True)
            else:
                cv = StratifiedKFold(n_splits=int(num_folds), shuffle=True)
            
            try:
                drug_pred = cross_val_predict(clf, drug_ge_table, drug_row, cv=cv,
                    n_jobs=24)
            except ValueError:
                continue

            # Get the actual scores of CV.
            for score_name in score_type_dct:
                key = (classifier_name, score_name, drug_name) # TODO key?
                assert key not in score_dct
                # TODO: stratified kfold for classifier?
                if data_source in ['gdsc', 'xeno']:
                    score = score_type_dct[score_name](drug_pred, drug_row)
                else:
                    score = score_type_dct[score_name](drug_pred, drug_row, average='weighted')
                # score_dct[key] = np.append(score_dct[key], score)
                score_dct[key] = score

    score_dct = pd.Series(score_dct).unstack()
    fname = '%s_%s_%s' % (data_source, method, num_folds)
    if method == 'essentiality':
        fname += '_%s' % input_net.split('/')[-1]
    score_dct.to_csv('./results/prediction_scores/%s.csv' % fname)

def main():
    generate_directories()
    global hgnc_to_ensg_dct
    hgnc_to_ensg_dct = preprocess_batch_effect.read_hgnc_mappings()

    # TODO: turning off warnings.
    warnings.filterwarnings('ignore')

    net_dir = 'data/CRISPR_networks/'
    net_dir = 'data/baseline_networks/'
    network_l = os.listdir(net_dir)
    data_l = ['xeno','tcga','gdsc']
    method_l = ['baseline']
    num_folds = 5
    xeno_type = 'Models'
    # for net in network_l:
    for net in ['net']: # TODO
        input_net = net_dir + net
        print('running network'+net)
        for data_source in data_l:
            for method in method_l:
                print(data_source, method)
                train_classifiers(input_net, data_source, method, num_folds, xeno_type)

if __name__ == '__main__':
    main()  