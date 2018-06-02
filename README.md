# xenograft_drp
Drug response prediction with xenograft data.

### Exploring batch effects.

1.  Obtain mappings from biomart. Get Human genes (GRCh38.p10). Check "Gene stable ID" and "HGNC" symbol. Check unique results only, export to tsv. Move to ./data/. Rename to hgnc_to_ensg.txt.

2.  Place GDSC data in ./data/GDSC.
    Place TCGA data in ./data/TCGA.
    Both data's drug response files need to be placed in tab-separated .txt files,
    since the script can't properly read .xlsx files.
    
    ```bash
    python preprocess_batch_effect.py [-h] -d {single,all} [-x {Models,Samples}]
    ```
    
3. Preprocess the KEGG pathways for Sparse-Group lasso.

4. Adding tissue types as binary features to the gene by sample matrix.

    ```bash
    python concatenate_sample_features.py
    ```

### GelNet

1. Run with different types of objective functions.

    ```bash
    Rscript gelnet_script.R {li, nick, traditional, d0}
    ```

### SGL

1. Cluster gene-gene networks.

    ```bash
    python cluster_networks_sgl.py
    ```
    
### Lasso tissue idea 1.

1. Run lasso on individual tissue types.

    ```bash
    python lasso_tissue.py
    ```

2. Runs idea 2 for individual tissue types, with a slight modification described in the script.

    ```bash
    python lasso_tissue_idea_2.py -i [1, 2]
    ```
3. The previous script also saves the weights for genes. Analyze these weights and print out the gene sets for each tissue-drug pair.

    ```bash
    python extract_gene_weights.py
    ```
    
    Outputs ./data/gdsc_tcga/idea_2_lasso_weights/nonzero_gene_lists.txt