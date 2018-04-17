### Author: Edward Huang

import community
import networkx as nx
import os
import subprocess

def read_network(fname):
    '''
    Processes a network to prepare for clustering.
    '''
    G = nx.Graph()
    f = open(fname, 'r')
    for line in f:
        try:
            gene_a, gene_b, weight = line.strip().split('\t')[:3]
        except ValueError:
            print line
            continue
        if 'ENSG' not in gene_a:
            return
        G.add_edge(gene_a, gene_b, weight=float(weight))
    f.close()
    return G

def read_gene_expr():
    '''
    Get the list of genes in order.
    '''
    gene_expr_lst = []
    f = open('./data/gdsc_tcga/tcga_expr_postCB.csv', 'r')
    header = f.readline()
    for line in f:
        gene = line.split(',')[0]
        gene_expr_lst += [gene]
    f.close()
    return gene_expr_lst

def write_mappings(final_mappings, fname):
    '''
    Writes out final mappings to the file.
    '''
    out = open('./data/KN_communities/%s' % fname, 'w')
    out.write(','.join(map(str, final_mappings)) + '\n')
    out.close()

def write_communities(fname):
    # TODO: Uncomment this next block.
    G = read_network('./data/KN/%s' % fname)
    if G == None: # Skip non-gene networks.
        return
    # part = community.best_partition(G)
    
    # gene_expr_lst = read_gene_expr()
    
    # # Map the list of genes to the list of mappings.
    # final_mappings = []
    # for gene in gene_expr_lst:
    #     if gene not in part:
    #         final_mappings += [0]
    #     else:
    #         final_mappings += [part[gene]+1]
    # assert len(final_mappings) == len(gene_expr_lst)
    
    # # Write out the final mappings.
    # write_mappings(final_mappings, fname)
    subprocess.call('Rscript sgl_script.R ./data/KN_communities/%s' % fname,
        shell=True)

def main():
    if not os.path.exists('./data/KN_communities'):
        os.makedirs('./data/KN_communities')
    
    for fname in os.listdir('./data/KN'):
        print fname
        write_communities(fname)
    
if __name__ == '__main__':
    main()