# xenograft_drp
Drug response prediction with xenograft data.

### Exploring batch effects.

1.  Obtain mappings from biomart. Get Human genes (GRCh38.p10). Check "Gene stable ID" and "HGNC" symbol. Check unique results only, export to tsv. Move to ./data/. Rename to hgnc_to_ensg.txt.

2.  Place GDSC data in ./data/GDSC.
    Place TCGA data in ./data/TCGA.
    Both data's drug response files need to be placed in tab-separated .txt files,
    since the script can't properly read .xlsx files.
    
    ```bash
    python preprocess_batch_effect.py [-d] [-x]
    ```