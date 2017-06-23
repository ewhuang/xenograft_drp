### Author: Edward Huang

import argparse
from csv import reader
import matplotlib
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
import subprocess

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab

tcga_folder = './data/TCGA'

def generate_directories():
    for subfolder in ('gdsc_tcga', 'gdsc_tcga_Samples', 'gdsc_tcga_Models'):
        for pattern in ('./data/%s', './data/%s/before_combat',
            './data/%s/after_combat', './data/%s/dr_matrices', './data/%s/single_drugs',
            './data/%s/single_drugs/before_combat', './data/%s/single_drugs/after_combat',
            './data/%s/single_drugs/batch_effect_plots'):
            folder = pattern % subfolder
            if not os.path.exists(folder):
                os.makedirs(folder)

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def read_tcga_drug_response():
    '''
    Reads the TCGA drug response. Responses are categories.
    For each drug, find the samples treated by that drug.
    Returns a drug x sample response matrix.
    '''
    def read_tcga_spreadsheet(tcga_id_to_fname_dct, fname):
        '''
        Returns a dictionary mapping the sample filenames to their TCGA sample IDs.
        Key: sample filename -> str, e.g., 010caf19-9f62-4488-a2b5-eacdb795d66e.FPKM.txt
        Value: sample ID -> str, e.g., TCGA-OR-A5KT-01
        '''
        f = open(fname, 'r')
        for sample_fname, sample_type, sample_id, race, cancer in reader(f):
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
    for subfolder in listdir_fullpath('%s/RNAseq' % tcga_folder):
        for fname in listdir_fullpath(subfolder):
            if 'Pheno_All' in fname:
                read_tcga_spreadsheet(tcga_id_to_fname_dct, fname)
    # Simple tests.
    assert tcga_id_to_fname_dct['TCGA-G9-7522'] == '3a8adea5-5520-41de-aed8-dfce2d52386e.FPKM.txt'
    assert tcga_id_to_fname_dct['TCGA-HU-A4GF'] == '1e46ba47-0a66-4f70-b508-c619c6129e33.FPKM.txt'

    # Read the drug response file.
    tcga_drug_to_sample_dct, tcga_dr_dct = {}, {}
    f = open('%s/drug_response.txt' % tcga_folder, 'r')
    f.readline() # Skip the header line.
    for line in f:
        line = line.strip().split('\t')
        sample_id, drug, response = line[1], line[2].lower(), line[4]
        # Skip samples that are not primary tumors.
        if sample_id not in tcga_id_to_fname_dct:
            continue
        # Map samples to filenames for the gene expression files.
        sample_fname = tcga_id_to_fname_dct[sample_id]
        assert '.txt' in sample_fname

        # Update the drug's related samples.
        if drug not in tcga_drug_to_sample_dct:
            tcga_drug_to_sample_dct[drug] = set([])
        tcga_drug_to_sample_dct[drug].add(sample_fname)

        # Update the drug response dictionary.
        tcga_dr_dct[(drug, sample_fname)] = response
    f.close()
    # Convert drug response dictionary to dataframe.
    tcga_dr_df = pd.Series(tcga_dr_dct).unstack()
    return tcga_drug_to_sample_dct, tcga_dr_df

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
        f.readline() # Skip the header line.
        for line in f:
            line = line.strip().split('\t')
            assert len(line) == 5
            drug_id, drug_name = line[:2]
            assert drug_id not in drug_id_to_name_dct
            drug_id_to_name_dct[drug_id] = drug_name.lower()
        f.close()
        return drug_id_to_name_dct

    drug_id_to_name_dct = read_drug_id_to_name_dct()
    # Simple tests.
    assert drug_id_to_name_dct['1'] == 'erlotinib'
    assert drug_id_to_name_dct['172'] == 'embelin'

    gdsc_drug_to_sample_dct, gdsc_dr_dct = {}, {}
    f = open('./data/GDSC/v17_fitted_dose_response.txt', 'r')
    f.readline() # Skip the header line.
    for line in f:
        line = line.strip().split('\t')
        sample_id, drug_id, ln_ic50 = line[2], line[3], line[5]
        assert sample_id.isdigit()
        # Skip unmappable drugs and the worse version of bicalutamide.
        if drug_id not in drug_id_to_name_dct or drug_id == '150':
            continue
        drug_name = drug_id_to_name_dct[drug_id]

        # Update the drug with the sample.
        if drug_name not in gdsc_drug_to_sample_dct:
            gdsc_drug_to_sample_dct[drug_name] = set([])
        gdsc_drug_to_sample_dct[drug_name].add(sample_id)

        # Update the drug response dictionary.
        gdsc_dr_dct[(drug_name, sample_id)] = float(ln_ic50)
    f.close()
    # Now, convert the drug response dictionary to a dataframe.
    gdsc_dr_df = pd.Series(gdsc_dr_dct).unstack()
    return gdsc_drug_to_sample_dct, gdsc_dr_df

def read_xeno_drug_response(xeno_type):
    '''
    Maps each drug to the cell lines that have response values for that drug.
    '''
    def read_gdsc_translation():
        '''
        Reads the drugs the Xenograft has in common with GDSC.
        '''
        drug_translation_dct = {}
        f = open('./data/Data_summary.txt', 'r')
        f.readline()
        for line in f:
            line = line.strip().split('\t')
            if len(line) != 3:
                continue
            drug, gdsc, xenograft = line
            # Skip drugs that do not exist for both datasets.
            if 'Y' not in gdsc or 'Y' not in xenograft:
                continue
            if '(' not in xenograft:
                drug_translation_dct[drug] = drug.lower()
            else:
                # [1:-1] to exclude the surrounding parentheses.
                drug_translation_dct[xenograft.split()[1][1:-1]] = drug.lower()
        f.close()
        return drug_translation_dct

    drug_translation_dct = read_gdsc_translation()
    # Simple tests.
    assert 'Cetuximab' not in drug_translation_dct
    assert 'cetuximab' not in drug_translation_dct
    assert drug_translation_dct['GSK2118436'] == 'dabrafenib'

    xeno_drug_to_sample_dct, xeno_dr_dct = {}, {}
    f = open('./data/Xenograft2/DrugResponsesAUC%s.txt' % xeno_type, 'r')
    f.readline() # Skip the header line.
    for line in f:
        line = line.strip().split('\t')
        if xeno_type == 'Models':
            sample, drug, ic50 = line[0], line[1], line[3]
        elif xeno_type == 'Samples':
            sample, drug, ic50 = line[0], line[2], line[4]
        assert '.txt' not in sample and not sample.isdigit()

        # Skip drugs that are not also in GDSC.
        if drug not in drug_translation_dct:
            continue
        drug = drug_translation_dct[drug]

        # Update the drug with the sample.
        if drug not in xeno_drug_to_sample_dct:
            xeno_drug_to_sample_dct[drug] = set([])
        xeno_drug_to_sample_dct[drug].add(sample)

        # Update the drug response dictionary.
        key = (drug, sample)
        assert key not in xeno_dr_dct
        xeno_dr_dct[key] = np.log(float(ic50))
    f.close()
    # Now, convert drug response dictionary to a dataframe.
    xeno_dr_df = pd.Series(xeno_dr_dct).unstack()
    return xeno_drug_to_sample_dct, xeno_dr_df

def read_tcga_gene_expr():
    def read_tcga_gene_expr_file(fname, gene_expr_table):
        '''
        Reads a gene x sample matrix. Contains many zeros.
        '''
        curr_table = pd.read_csv(fname, index_col=0)
        ret = pd.concat([gene_expr_table, curr_table], axis=1)
        return ret
    # Concatenate the gene expression matrices across the different cancers.
    tcga_table = pd.DataFrame()
    for subfolder in listdir_fullpath('%s/RNAseq' % tcga_folder):
        for fname in listdir_fullpath(subfolder):
            if 'FPKM' in fname:
                tcga_table = read_tcga_gene_expr_file(fname, tcga_table)
    # Convert to numeric.
    tcga_table = tcga_table.apply(pd.to_numeric)
    # Remove NaNs.
    tcga_table = tcga_table.dropna(axis=0, how='any')
    # At least 10% of genes must have value > 1 pre log-transformation.
    tcga_table = tcga_table.drop([idx for idx in tcga_table.index if (
        tcga_table.loc[idx]<1).astype(int).sum() > tcga_table.shape[1]*0.9], axis=0)
    # Add pseudo-counts to avoid NaN errors with log.
    tcga_table = tcga_table.add(0.1)
    # Log 2 transform just the TCGA matrix.
    return np.log2(tcga_table)

def read_gdsc_gene_expr():
    '''
    Returns a dataframe.
    '''
    gdsc_table = pd.read_table('./data/GDSC/sanger1018_brainarray_ensemblgene_rma.txt',
        index_col=0, header=0)
    # Convert to numeric.
    gdsc_table = gdsc_table.apply(pd.to_numeric)
    # Remove NaNs.
    gdsc_table = gdsc_table.dropna(axis=0, how='any')
    return gdsc_table

def read_hgnc_mappings():
    hgnc_to_ensg_dct = {}
    f = open('./data/hgnc_to_ensg.txt','r')
    f.readline()
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

def read_xeno_gene_expr(xeno_type):
    '''
    Returns a dataframe.
    '''
    hgnc_to_ensg_dct = read_hgnc_mappings()
    # Read the gene expression file.
    xeno_expr_matrix, gene_list = [], []
    f = open('./data/Xenograft2/Expression%s.txt' % xeno_type, 'r')
    for i, line in enumerate(f):
        line = line.split()
        if i == 0: # Process header line.
            sample_list = line[1:]
            continue
        hgnc_id, expr_lst = line[0], line[1:]
        # Skip genes that have NA in them.
        if hgnc_id not in hgnc_to_ensg_dct or 'NA' in expr_lst:
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
    # Convert to numeric.
    xeno_table = xeno_table.apply(pd.to_numeric)
    return xeno_table

def concatenate_expression_matrices(curr_drug, drugs_in_common=[]):
    '''
    Concatenates two gene expression matrices from different sources, and then
    writes out to file.
    '''
    # Slice only the samples that have taken the current drug.
    if curr_drug == 'all':
        # For all drugs, get all samples for all drug keys.
        tcga_drug_samples, gdsc_drug_samples = set([]), set([])
        for drug in drugs_in_common:
            tcga_drug_samples = tcga_drug_samples.union(tcga_drug_to_sample_dct[drug])
            gdsc_drug_samples = gdsc_drug_samples.union(gdsc_drug_to_sample_dct[drug])
        # Get the Xenograft gene expression dataframe, if necessary.
        if args.xeno_type != None:
            xeno_drug_samples = set([])
            for drug in drugs_in_common:
                xeno_drug_samples = xeno_drug_samples.union(xeno_drug_to_sample_dct[drug])
    else:
        # For single drugs, get just the samples for the current drug.
        tcga_drug_samples = tcga_drug_to_sample_dct[curr_drug]
        gdsc_drug_samples = gdsc_drug_to_sample_dct[curr_drug]
        if args.xeno_type != None:
            xeno_drug_samples = xeno_drug_to_sample_dct[curr_drug]

    # Slice the tables based on the samples for the current drug.
    tcga_samples = list(tcga_drug_samples.intersection(list(tcga_table)))
    drug_tcga_table = tcga_table[tcga_samples]

    gdsc_samples = list(gdsc_drug_samples.intersection(list(gdsc_table)))
    drug_gdsc_table = gdsc_table[gdsc_samples]
    
    concat_table = [drug_tcga_table, drug_gdsc_table]

    if args.xeno_type != None:
        xeno_samples = list(xeno_drug_samples.intersection(list(xeno_table)))
        drug_xeno_table = xeno_table[xeno_samples]
        concat_table += [drug_xeno_table]

    # Concatenate the TCGA and GDSC tables.
    gene_expr_table = pd.concat(concat_table, axis=1)
    # Drop a gene if there are any NaNs.
    gene_expr_table = gene_expr_table.dropna(axis=0, how='any')
    # Drop rows with low standard deviation.
    gene_expr_table = gene_expr_table[gene_expr_table.std(axis=1)>0.1]
    # Write the full table out to file.
    gene_expr_table.to_csv('./data/%s/before_combat/%s_gene_expr_before_combat.csv'
        % (results_folder, curr_drug))

    if args.drug_strat == 'all':
        if args.xeno_type == None:
            return (tcga_samples, gdsc_samples)
        else:
            return (tcga_samples, gdsc_samples, xeno_samples)

def write_drug_response_matrices(drugs_in_common, df_tuple_lst, sample_lst):
    '''
    Write out the drug response matrices. Drop drugs that are not in
    drugs_in_common.
    '''
    for i, (dr_df, fname) in enumerate(df_tuple_lst):
        # Slice the samples that are in common between drug response and gene expression.
        dr_df = dr_df[sample_lst[i]]
        # Drop drugs that aren't in the combined datasets.
        for drug in dr_df.index.values:
            # Split by underscore because we tagged on drug IDs for GDSC.
            if drug not in drugs_in_common:
                dr_df = dr_df.drop(drug)
        # Remove missing columns.
        dr_df.dropna(axis=1, how='all', inplace=True)
        # Fill in missing values with easy-to-identify value, 'NA'.
        dr_df.fillna(value='NA', inplace=True)
        dr_df.to_csv('./data/%s/dr_matrices/%s.csv' % (results_folder, fname))

def write_drug_matrices():
    ### First, get the drug response information, and the samples that are
    ### treated by each drug for each dataset.
    global tcga_drug_to_sample_dct
    # Get TCGA information.
    tcga_drug_to_sample_dct, tcga_dr_df = read_tcga_drug_response()
    tcga_drugs = tcga_drug_to_sample_dct.keys()
    # Get GDSC information.
    global gdsc_drug_to_sample_dct
    gdsc_drug_to_sample_dct, gdsc_dr_df = read_gdsc_drug_response()
    gdsc_drugs = gdsc_drug_to_sample_dct.keys()

    # Get the drugs that are shared by both datasets.
    drugs_in_common = set(tcga_drugs).intersection(gdsc_drugs)
    # Write out drug response matrices when using all drugs.
    if args.drug_strat == 'all':
        df_tuple_lst = [(tcga_dr_df, 'tcga_dr'), (gdsc_dr_df, 'gdsc_dr')]

    # Get the Xenograft information, if required.
    if args.xeno_type != None:
        global xeno_drug_to_sample_dct
        xeno_drug_to_sample_dct, xeno_dr_df = read_xeno_drug_response(args.xeno_type)
        xeno_drugs = xeno_drug_to_sample_dct.keys()
        # Further intersect drugs with Xenograft drugs.
        drugs_in_common = drugs_in_common.intersection(xeno_drugs)
        if args.drug_strat == 'all':
            df_tuple_lst += [(xeno_dr_df, 'xeno_dr_%s' % args.xeno_type)]

    ### Next, get the gene expression information.
    # Get the TCGA gene expression dataframe.
    global tcga_table
    tcga_table = read_tcga_gene_expr()
    # Get the GDSC gene expression dataframe.
    global gdsc_table
    gdsc_table = read_gdsc_gene_expr()
    # Get the Xenograft gene expression dataframe, if required.
    if args.xeno_type != None:
        global xeno_table
        xeno_table = read_xeno_gene_expr(args.xeno_type)

    # Turn drugs into a list for ordering.
    drugs_in_common = list(drugs_in_common)

    if args.drug_strat == 'single':
        pool = Pool(processes=20)
        pool.map(concatenate_expression_matrices, drugs_in_common)
    elif args.drug_strat == 'all':
        sample_lst = concatenate_expression_matrices('all', drugs_in_common)
        write_drug_response_matrices(drugs_in_common, df_tuple_lst, sample_lst)

    return drugs_in_common

def write_pheno_file(drug):
    f = open('./data/%s/before_combat/%s_gene_expr_before_combat.csv' %
        (results_folder, drug), 'r')
    # Read the header of the gene expression matrix, given a drug.
    # This is the list of samples.
    header = next(reader(f))[1:]
    f.close()
    # Write out the batches of each drug.
    out = open('./data/%s/before_combat/%s_pheno.txt' % (results_folder, drug), 'w')
    out.write('\tsample\tbatch\n')
    for i, sample in enumerate(header):
        sample_num = i + 1
        if '.txt' in sample: # TCGA
            batch_num = 1
        elif sample.isdigit(): # GDSC
            batch_num = 2
        else:
            batch_num = 3 # Xenograft
        out.write('%s\t%d\t%d\n' % (sample, sample_num, batch_num))
    out.close()

def plot_stitched_gene_expr(drug, when):
    assert when in ('before', 'after')
    mat = []
    f = open('./data/%s/%s_combat/%s_gene_expr_%s_combat.csv' % (results_folder,
        when, drug, when), 'r')
    it = reader(f)
    header = next(it)[1:]
    # Remove initial X placed there by ComBat.
    header = [col[1:] if col[0] == 'X' else col for col in header]
    colors = []
    for sample in header:
        if '.txt' in sample:
            colors += ['blue']
        elif sample.isdigit():
            colors += ['green']
        else:
            colors += ['black']
    for line in it:
        # Skip lines that are all NAs.
        if line[1:] == ['NA'] * len(header):
            continue
        # line = map(float, [0 if expr == '' else expr for expr in line[1:]])
        line = map(float, line[1:])
        assert len(colors) == len(line)
        mat += [line]
    f.close()

    if mat == []:
        return

    # Reduce to two dimensions.
    dim_reduc = PCA(n_components=2)
    mat = dim_reduc.fit_transform(np.array(mat).T)

    # plot the dimensions.
    x_points = [point[0] for point in mat]
    y_points = [point[1] for point in mat]
    # Plot resulting feature matrix.
    plt.scatter(x=x_points, y=y_points, c=colors, s=20, edgecolors='black')
    plt.show()

    pylab.savefig('./data/%s/batch_effect_plots/%s_%s_combat.png' %
        (results_folder, drug, when))
    plt.close()

def separate_concat_mats():
    '''
    Finally, separates the concatenated matrix into its constituent parts after
    running ComBat.
    '''
    table = pd.read_csv('./data/%s/after_combat/all_gene_expr_after_combat.csv' %
        results_folder, index_col=0)
    current_columns = list(table)
    total_num_columns = 0 # Sanity check.
    # Write out the TCGA table.
    tcga_cols = [col for col in current_columns if '.txt' in col]
    tcga_table = table[tcga_cols]
    tcga_table.to_csv('./data/%s/tcga_expr_postCB.csv' % results_folder)
    total_num_columns += len(list(tcga_table))
    # Write out the GDSC table.
    gdsc_cols = [col for col in current_columns if col[1:].isdigit()]
    gdsc_table = table[gdsc_cols]
    gdsc_table.to_csv('./data/%s/gdsc_expr_postCB.csv' % results_folder)
    total_num_columns += len(list(gdsc_table))

    if args.xeno_type != None:
        xeno_cols = [col for col in current_columns if ('.txt' not in col and
            not col[1:].isdigit())]
        xeno_table = table[xeno_cols]
        total_num_columns += len(list(xeno_table))
        xeno_table.to_csv('./data/%s/xeno_%s_expr_postCB.csv' % (results_folder, args.xeno_type))
    assert total_num_columns == len(current_columns)

def parse_args():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--drug_strat', help='whether or not to combine drug by drug',
        choices=['single', 'all'], required=True)
    parser.add_argument('-x', '--xeno_type', help='type of xenograft data',
        choices=['Models', 'Samples'])
    args = parser.parse_args()

def parse_results_folder():
    global results_folder
    if args.xeno_type != None:
        results_folder = 'gdsc_tcga_%s' % args.xeno_type
    else:
        results_folder = 'gdsc_tcga'
    if args.drug_strat == 'single':
        results_folder += '/single_drugs'

def main():
    parse_args()
    parse_results_folder()
    generate_directories()

    drugs_in_common = write_drug_matrices()

    if args.drug_strat == 'all':
        drugs_in_common = ['all']

    for drug in drugs_in_common:
        write_pheno_file(drug)

    command = ('Rscript combat_batch_script.R %s %s' % (results_folder,
        ' '.join(drugs_in_common)))
    subprocess.call(command, shell=True)

    if args.drug_strat == 'single':
        for drug in drugs_in_common:
            plot_stitched_gene_expr(drug, 'before')
            plot_stitched_gene_expr(drug, 'after')
    else:
        separate_concat_mats()

if __name__ == '__main__':
    main()