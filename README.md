# xenograft_drp
Drug response prediction with xenograft data.

### Exploring batch effects.

1.  Place GDSC data in ./data/GDSC.
    Place TCGA data in ./data/TCGA.
    Both data's drug response files need to be placed in tab-separated .txt files,
    since the script can't properly read .xlsx files.
    
    ```bash
    python preprocess_batch_effect.py
    ```