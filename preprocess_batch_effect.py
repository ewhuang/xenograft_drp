### Author: Edward Huang

from csv import reader
import matplotlib
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import subprocess

matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import pylab

tcga_folder = './data/TCGA'

def generate_directories():
    plot_folder = './data/batch_effect_plots'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

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
        if sample_type != 'Primary Tumor':
            continue
        sample_id = '-'.join(sample_id.split('-')[:-1])
        assert sample_id not in tcga_id_to_fname_dct
        tcga_id_to_fname_dct[sample_id] = sample_fname
    f.close()

def read_tcga_drug_response(tcga_id_to_fname_dct):
    '''
    Reads the TCGA drug response.
    '''
    tcga_drug_to_sample_dct = {}
    f = open('%s/drug_response.txt' % tcga_folder, 'r')
    f.readline()
    for line in f:
        line = line.strip().split('\t')
        sample_id, drug, response = line[1], line[2].lower(), line[4]
        # Skip samples that are not primary tumors.
        if sample_id not in tcga_id_to_fname_dct:
            continue
        # Update the output dictionary.
        if drug not in tcga_drug_to_sample_dct:
            tcga_drug_to_sample_dct[drug] = set([])
        sample_fname = tcga_id_to_fname_dct[sample_id]
        tcga_drug_to_sample_dct[drug].add(sample_fname)
    f.close()
    return tcga_drug_to_sample_dct

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def read_tcga_gene_expr(fname, drug_samples, gene_expr_table):
    table = pd.read_csv(fname, index_col=0)
    current_columns = list(table)
    # Get the samples for this cancer's gene expression.
    samples_in_common = list(drug_samples.intersection(current_columns))
    if len(samples_in_common) == 0:
        return gene_expr_table
    table = table[samples_in_common]
    ret = pd.concat([gene_expr_table, table], axis=1)
    assert list(table.index) == list(ret.index)
    return ret

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

def read_gdsc_drug_response():
    '''
    Maps each drug to the cell lines that have response values for that drug.
    '''
    drug_id_to_name_dct = read_drug_id_to_name_dct()

    gdsc_drug_to_sample_dct = {}
    f = open('./data/GDSC/v17_fitted_dose_response.txt', 'r')
    f.readline()
    for line in f:
        line = line.strip().split('\t')
        cosmic_id, drug_id = line[2], line[3]
        if drug_id not in drug_id_to_name_dct:
            continue
        drug_name = drug_id_to_name_dct[drug_id]
        if drug_name not in gdsc_drug_to_sample_dct:
            gdsc_drug_to_sample_dct[drug_name] = set([])
        gdsc_drug_to_sample_dct[drug_name].add(cosmic_id)
    f.close()
    return gdsc_drug_to_sample_dct

def read_gdsc_gene_expr(sample_set):
    '''
    Returns a dataframe.
    '''
    gdsc_drug_response = pd.read_table(
        './data/GDSC/sanger1018_brainarray_ensemblgene_rma.txt', index_col=0,
        header=0)
    gene_expr_samples = list(gdsc_drug_response)
    sample_list = list(sample_set.intersection(gene_expr_samples))
    return gdsc_drug_response[sample_list]

def write_drug_matrices():
    # Get TCGA information.
    tcga_id_to_fname_dct = {}
    for subfolder in listdir_fullpath('%s/RNAseq' % tcga_folder):
        for fname in listdir_fullpath(subfolder):
            if 'Pheno_All' not in fname:
                continue
            read_tcga_spreadsheet(tcga_id_to_fname_dct, fname)

    tcga_drug_to_sample_dct = read_tcga_drug_response(tcga_id_to_fname_dct)

    # Get GDSC information.
    gdsc_drug_to_sample_dct = read_gdsc_drug_response()
    drugs_in_common = list(set(gdsc_drug_to_sample_dct.keys()).intersection(
        tcga_drug_to_sample_dct.keys()))

    out = open('./data/min_max_values.tsv', 'w')
    out.write('\tTCGA max\tTCGA min\tGDSC max\tGDSC min\n')
    for curr_drug in drugs_in_common:
    # for curr_drug in ['cisplatin']:
        print curr_drug
        tcga_drug_samples = tcga_drug_to_sample_dct[curr_drug]

        tcga_table = pd.DataFrame()
        # Get the TCGA gene expression dataframe.
        for subfolder in listdir_fullpath('%s/RNAseq' % tcga_folder):
            for fname in listdir_fullpath(subfolder):
                if 'FPKM' not in fname:
                    continue
                tcga_table = read_tcga_gene_expr(fname, tcga_drug_samples, tcga_table)               
        # Log 2 transform.
        tcga_table = tcga_table.add(0.1)
        tcga_table = np.log2(tcga_table)

        # Get the GDSC gene expression dataframe.
        gdsc_drug_samples = gdsc_drug_to_sample_dct[curr_drug]
        gdsc_table = read_gdsc_gene_expr(gdsc_drug_samples)
        # Concatenate the TCGA and GDSC tables.
        gene_expr_table = pd.concat([tcga_table, gdsc_table], axis=1)
        # Replace NaNs with 0s.
        # gene_expr_table = gene_expr_table.fillna(value=0) # TODO: currently filling.
        gene_expr_table = gene_expr_table.dropna(axis=0, how='any')
        # Drop rows with low standard deviation.
        gene_expr_table = gene_expr_table[gene_expr_table.std(axis=1)>0.1]
        # Get min/max values.
        tcga_cols = [x for x in gene_expr_table.columns if '.txt' in x]
        tcga_vals = gene_expr_table[tcga_cols].values
        out.write('%s\t%s\t%s\t' % (curr_drug, tcga_vals.max(), tcga_vals.min()))
        gdsc_cols = [x for x in gene_expr_table.columns if '.txt' not in x]
        gdsc_vals = gene_expr_table[gdsc_cols].values
        out.write('%s\t%s\n' % (gdsc_vals.max(), gdsc_vals.min()))
        # Write the full table out to file.
        gene_expr_table.to_csv('./data/%s_gene_expr_before_combat.csv' % curr_drug)
    out.close()
    return drugs_in_common

def write_pheno_file(drug, header):
    out = open('./data/%s_pheno.txt' % drug, 'w')
    out.write('\tsample\tbatch\n')
    for i, sample in enumerate(header):
        if '.txt' in sample:
            batch = 1
        else:
            batch = 2
        out.write('%s\t%d\t%d\n' % (sample, i+1, batch))
    out.close()

def plot_stitched_gene_expr(drug, when):
    assert when in ('before', 'after')
    mat = []
    f = open('./data/%s_gene_expr_%s_combat.csv' % (drug, when), 'r')
    it = reader(f)
    header = it.next()[1:]
    write_pheno_file(drug, header)
    colors = ['black' if '.txt' in sample else 'white' for sample in header]
    for line in it:
        # Skip lines that are all NAs.
        if line[1:] == ['NA'] * len(header):
            continue
        line = map(float, [0 if expr == '' else expr for expr in line[1:]])
        assert len(colors) == len(line)
        mat += [line]
    f.close()

    if mat == []:
        return

    # Normalize the matrix.
    # mat = normalize(mat, norm='l2', axis=0)

    # Reduce to two dimensions.
    dim_reduc = PCA(n_components=2)
    mat = dim_reduc.fit_transform(np.array(mat).T)

    # plot the dimensions.
    x_points = [point[0] for point in mat]
    y_points = [point[1] for point in mat]
    # Plot resulting feature matrix.
    plt.scatter(x=x_points, y=y_points, c=colors, s=20)
    plt.show()

    pylab.savefig('./data/batch_effect_plots/%s_%s_combat.png' % (drug, when))
    plt.close()

def main():
    generate_directories()

    drugs_in_common = write_drug_matrices()

    command = ('Rscript combat_batch_script.R %s' % ' '.join(drugs_in_common))
    print command
    subprocess.call(command, shell=True)

    for drug in drugs_in_common:
        print drug
        plot_stitched_gene_expr(drug, 'before')
        plot_stitched_gene_expr(drug, 'after')

if __name__ == '__main__':
    main()