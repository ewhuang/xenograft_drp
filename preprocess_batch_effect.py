### Author: Edward Huang

from csv import reader
import matplotlib
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import subprocess
import sys

matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import pylab

tcga_folder = './data/TCGA'

def generate_directories():
    for folder in ('./data/gdsc_tcga', './data/gdsc_tcga_xeno',
        './data/gdsc_tcga/before_combat', './data/gdsc_tcga_xeno/before_combat',
        './data/gdsc_tcga/after_combat', './data/gdsc_tcga_xeno/after_combat',
        './data/gdsc_tcga/dr_matrices', './data/gdsc_tcga_xeno/dr_matrices',
        './data/gdsc_tcga/batch_effect_plots', './data/gdsc_tcga_xeno/batch_effect_plots'):
        if not os.path.exists(folder):
            os.makedirs(folder)

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

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
    for subfolder in listdir_fullpath('%s/RNAseq' % tcga_folder):
        for fname in listdir_fullpath(subfolder):
            if 'Pheno_All' in fname:
                read_tcga_spreadsheet(tcga_id_to_fname_dct, fname)

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

    gdsc_drug_to_sample_dct, gdsc_dr_dct = {}, {}
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

        # Update the drug with the sample.
        if drug_name not in gdsc_drug_to_sample_dct:
            gdsc_drug_to_sample_dct[drug_name] = set([])
        gdsc_drug_to_sample_dct[drug_name].add(sample_id)

        # Update the drug response dictionary.
        key = ('%s_%s' % (drug_name, drug_id), sample_id)
        assert key not in gdsc_dr_dct
        gdsc_dr_dct[key] = float(ln_ic50)
    f.close()
    # Now, convert the drug response dictionary to a dataframe.
    gdsc_dr_df = pd.Series(gdsc_dr_dct).unstack()
    return gdsc_drug_to_sample_dct, gdsc_dr_df

def read_xeno_drug_response():
    '''
    Maps each drug to the cell lines that have response values for that drug.
    '''
    xeno_drug_to_sample_dct, xeno_dr_dct = {}, {}
    f = open('./data/Xenograft2/DrugResponsesAUC%s.txt' % xeno_type, 'r')
    f.readline() # Skip the header line.
    for line in f:
        line = line.strip().split('\t')
        if xeno_type == 'Models':
            sample, drug, ic50 = line[0], line[1].lower(), np.log(float(line[3]))
        elif xeno_type == 'Samples':
            sample, drug, ic50 = line[0], line[2].lower(), np.log(float(line[4]))
        assert '.txt' not in sample and not sample.isdigit()
        # Update the drug with the sample.
        if drug not in xeno_drug_to_sample_dct:
            xeno_drug_to_sample_dct[drug] = set([])
        xeno_drug_to_sample_dct[drug].add(sample)

        # Update the drug response dictionary.
        key = (drug, sample)
        assert key not in xeno_dr_dct
        xeno_dr_dct[key] = ic50
    f.close()
    # Now, convert drug response dictionary to a dataframe.
    xeno_dr_df = pd.Series(xeno_dr_dct).unstack()
    return xeno_drug_to_sample_dct, xeno_dr_df

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

def read_gdsc_gene_expr(drug_samples):
    '''
    Returns a dataframe.
    '''
    gdsc_table = pd.read_table(
        './data/GDSC/sanger1018_brainarray_ensemblgene_rma.txt', index_col=0,
        header=0)
    gene_expr_samples = list(gdsc_table) # Get the name of the colums.
    sample_list = list(drug_samples.intersection(gene_expr_samples))
    return gdsc_table[sample_list]

def read_xeno_gene_expr(drug_samples):
    '''
    Returns a dataframe.
    '''
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

def concatenate_expr_matrix_all_drugs(tcga_samples, gdsc_samples):
    '''
    Creates the master gene expression file.
    '''
    tcga_table = pd.DataFrame()
    for subfolder in listdir_fullpath('%s/RNAseq' % tcga_folder):
        for fname in listdir_fullpath(subfolder):
            if 'FPKM' in fname:
                tcga_table = read_tcga_gene_expr(fname, set(tcga_samples), tcga_table)
    # Log 2 transform just the TCGA matrix.
    tcga_table = tcga_table.add(0.1) # Add pseudo-counts to avoid NaN errors.
    tcga_table = np.log2(tcga_table)
    print tcga_table.shape

    gdsc_table = read_gdsc_gene_expr(set(gdsc_samples))
    exit()

def concatenate_expression_matrices(curr_drug):
    '''
    Concatenates two gene expression matrices from different sources, and then
    writes out to file.
    '''
    # # Get the TCGA gene expression dataframe.
    # tcga_table = pd.DataFrame()
    # tcga_drug_samples = tcga_drug_to_sample_dct[curr_drug]
    # for subfolder in listdir_fullpath('%s/RNAseq' % tcga_folder):
    #     for fname in listdir_fullpath(subfolder):
    #         if 'FPKM' in fname:
    #             tcga_table = read_tcga_gene_expr(fname, tcga_drug_samples, tcga_table)
    # # Log 2 transform just the TCGA matrix.
    # tcga_table = tcga_table.add(0.1) # Add pseudo-counts to avoid NaN errors.
    # tcga_table = np.log2(tcga_table)

    # Get the GDSC gene expression dataframe.
    gdsc_drug_samples = gdsc_drug_to_sample_dct[curr_drug]
    gdsc_table = read_gdsc_gene_expr(gdsc_drug_samples)
    print gdsc_table.shape
    exit()

    concat_table = [tcga_table, gdsc_table]
    folder = './data/gdsc_tcga/before_combat'

    # Get the Xenograft gene expression dataframe, if necessary.
    if hasXeno:
        xeno_drug_samples = xeno_drug_to_sample_dct[curr_drug]
        xeno_table = read_xeno_gene_expr(xeno_drug_samples)
        concat_table += [xeno_table]
        folder = './data/gdsc_tcga_xeno/before_combat'

    # Concatenate the TCGA and GDSC tables.
    gene_expr_table = pd.concat(concat_table, axis=1)
    # Drop NaN columns.
    gene_expr_table = gene_expr_table.dropna(axis=0, how='any')
    # Drop rows with low standard deviation.
    gene_expr_table = gene_expr_table[gene_expr_table.std(axis=1)>0.1]
    # TODO: drop drows with non-expressed genes.
    # Write the full table out to file.
    gene_expr_table.to_csv('%s/%s_gene_expr_before_combat.csv' % (folder,
        curr_drug))

    # # Get min/max values.
    # tcga_cols = [x for x in gene_expr_table.columns if '.txt' in x]
    # tcga_vals = gene_expr_table[tcga_cols].values
    # gdsc_cols = [x for x in gene_expr_table.columns if '.txt' not in x]
    # gdsc_vals = gene_expr_table[gdsc_cols].values
    # # Return the min/max values for each data source.
    # return (curr_drug, tcga_vals.max(), tcga_vals.min(), gdsc_vals.max(), gdsc_vals.min())

def write_drug_matrices():
    global tcga_drug_to_sample_dct, gdsc_drug_to_sample_dct

    # Get TCGA information.
    tcga_drug_to_sample_dct, tcga_dr_df = read_tcga_drug_response()
    tcga_drugs = tcga_drug_to_sample_dct.keys()
    # Get GDSC information.
    gdsc_drug_to_sample_dct, gdsc_dr_df = read_gdsc_drug_response()
    gdsc_drugs = gdsc_drug_to_sample_dct.keys()

    # Get the drugs that are shared by both datasets.
    drugs_in_common = set(tcga_drugs).intersection(gdsc_drugs)

    df_tuple_lst = [(tcga_dr_df, 'tcga_dr'), (gdsc_dr_df, 'gdsc_dr')]
    dr_folder = './data/gdsc_tcga/dr_matrices'

    # Get the Xenograft information, if desired.
    if hasXeno: # Global variable based on command line.
        global xeno_drug_to_sample_dct
        xeno_drug_to_sample_dct, xeno_dr_df = read_xeno_drug_response()
        xeno_drugs = xeno_drug_to_sample_dct.keys()
        # Further intersect drugs with Xenograft drugs.
        drugs_in_common = drugs_in_common.intersection(xeno_drugs)
        # Add Xenograft drug response dataframe.
        df_tuple_lst += [(xeno_dr_df, 'xeno_dr_%s' % xeno_type)]
        dr_folder = './data/gdsc_tcga_xeno/dr_matrices'

    # Write out the drug response matrices.
    for dr_df, fname in df_tuple_lst:
        for drug in dr_df.index.values:
            if drug.split('_')[0] not in drugs_in_common:
                dr_df.drop(drug, inplace=True)
        dr_df.dropna(axis=1, how='all', inplace=True)
        dr_df.fillna(value='NA', inplace=True)
        dr_df.to_csv('%s/%s.csv' % (dr_folder, fname))

    # Turn drugs into a list for ordering.
    drugs_in_common = list(drugs_in_common)

    concatenate_expr_matrix_all_drugs(list(tcga_dr_df), list(gdsc_dr_df))

    concatenate_expression_matrices('doxorubicin')
    exit()
    # TODO: uncomment.
    pool = Pool(processes=20)
    pool.map(concatenate_expression_matrices, drugs_in_common)
    # results = pool.map(concatenate_expression_matrices, drugs_in_common)

    # # Writing out the min/max values.
    # out = open('./data/gdsc_tcga/before_combat/min_max_values.tsv', 'w')
    # out.write('\tTCGA max\tTCGA min\tGDSC max\tGDSC min\n')
    # for tup in results:
    #     tup = map(str, results)
    #     out.write('%s\n' % tup)
    # out.close()

    return drugs_in_common

def write_pheno_file(drug):
    if hasXeno:
        folder = './data/gdsc_tcga_xeno/before_combat'
    else:
        folder = './data/gdsc_tcga/before_combat'
    f = open('%s/%s_gene_expr_before_combat.csv' % (folder, drug), 'r')
    it = reader(f)
    header = it.next()[1:]

    out = open('%s/%s_pheno.txt' % (folder, drug), 'w')
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
    if hasXeno:
        folder = './data/gdsc_tcga_xeno'
    else:
        folder = './data/gdsc_tcga'
    f = open('%s/%s_combat/%s_gene_expr_%s_combat.csv' % (folder, when, drug,
        when), 'r')
    it = reader(f)
    header = it.next()[1:]
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

    # Reduce to two dimensions.
    dim_reduc = PCA(n_components=2)
    mat = dim_reduc.fit_transform(np.array(mat).T)

    # plot the dimensions.
    x_points = [point[0] for point in mat]
    y_points = [point[1] for point in mat]
    # Plot resulting feature matrix.
    plt.scatter(x=x_points, y=y_points, c=colors, s=20)
    plt.show()

    pylab.savefig('%s/batch_effect_plots/%s_%s_combat.png' % (folder, drug, when))
    plt.close()

def separate_concat_mats(drugs_in_common):
    '''
    Finally, separates the concatenated matrix into its constituent parts after
    running ComBat.
    '''
    if hasXeno:
        expr_folder = './data/gdsc_tcga_xeno/after_combat'
    else:
        expr_folder = './data/gdsc_tcga/after_combat'        
    for drug in drugs_in_common:
        table = pd.read_csv('%s/%s_gene_expr_after_combat.csv' % (expr_folder,
            drug))
        current_columns = list(table)
        # Write out the TCGA table.
        tcga_cols = [col for col in current_columns if '.txt' in col]
        tcga_table = table[tcga_cols]
        tcga_table.to_csv('%s/%s_tcga.csv' % (expr_folder, drug))
        # Write out the GDSC table.
        gdsc_cols = [col for col in current_columns if '.txt' not in col]
        gdsc_table = table[gdsc_cols]
        gdsc_table.to_csv('%s/%s_gdsc.csv' % (expr_folder, drug))

def main():
    if len(sys.argv) not in [1, 2]:
        print 'Usage: python %s [Samples, Models]' % sys.argv[0]
        exit()
    global hasXeno
    hasXeno = False
    if len(sys.argv) == 2:
        global xeno_type
        xeno_type, hasXeno = sys.argv[1], True
        assert xeno_type in ['Samples', 'Models']

    generate_directories()

    drugs_in_common = write_drug_matrices()
    exit()

    for drug in drugs_in_common:
        write_pheno_file(drug)

    if hasXeno:
        folder = 'gdsc_tcga_xeno'
    else:
        folder = 'gdsc_tcga'
    command = ('Rscript combat_batch_script.R %s %s' % (folder,
        ' '.join(drugs_in_common)))
    subprocess.call(command, shell=True)

    for drug in drugs_in_common:
        plot_stitched_gene_expr(drug, 'before')
        plot_stitched_gene_expr(drug, 'after')

    separate_concat_mats(drugs_in_common)

if __name__ == '__main__':
    main()