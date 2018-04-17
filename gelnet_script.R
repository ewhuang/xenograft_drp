### Author: Edward Huang

library(gelnet)

change.files <- function(method_type) {

    gdsc_ge <- t(read.table('./data/gdsc_tcga/gdsc_expr_postCB.csv', header=TRUE, row.names=1, sep=',',
    check.names=FALSE))
    gdsc_dr <- t(read.table('./data/gdsc_tcga/dr_matrices/gdsc_dr.csv', header=TRUE, row.names=1, sep=',', na.strings = "NA",
    check.names=FALSE))
    
    tcga_ge <- t(read.table('./data/gdsc_tcga/tcga_expr_postCB.csv', header=TRUE, row.names=1, sep=',',
    check.names=FALSE))
    tcga_dr <- t(read.table('./data/gdsc_tcga/dr_matrices/tcga_dr.csv', header=TRUE, row.names=1, sep=',', na.strings="NA",
    check.names=FALSE))
    
    # Get the PPI into a feature-feature relationship matrix.
    ppi_mat <- read.table('./data/gdsc_tcga/STRING_experimental_KN02_table.csv', header=TRUE, row.names=1, sep=',',
    check.names=FALSE)
    print(dim(ppi_mat))
    penalty <- adj2nlapl(ppi_mat)
    
    # Construct the empty prediction matrix.
    pred_df <- data.frame(matrix(ncol = dim(tcga_ge)[1], nrow = 0))
    x <- rownames(tcga_ge)
    colnames(pred_df) <- x
    # y <- rownames(gdsc_dr)
    # row.names(pred_df) <- y
    
    drugs <- colnames(gdsc_dr)
    for(i in 1:ncol(gdsc_dr)) {
        row <- gdsc_dr[,i]
        
        good_indices <- which(!is.na(row))
        row <- as.numeric(row[good_indices])
        # print(row)
        # TODO: Change the parameters here in order to reate other methods.
        if (method_type == 'li') {
            gel <- gelnet.cv(gdsc_ge[good_indices,], row, 5, 5, nFolds=5, d=rep(1, nrow(penalty)), P = penalty)
        } else if (method_type == 'nick') {
            gel <- gelnet.cv(gdsc_ge[good_indices,], row, 5, 5, nFolds=5, d=rep(1e-100, ncol(penalty)), P = diag(nrow(penalty)) + penalty * 0.5)
        } else if (method_type == 'd0') {
            gel <- gelnet.cv(gdsc_ge[good_indices,], row, 5, 5, nFolds=5, d=rep(1e-100, nrow(penalty)), P=penalty)
        } else if (method_type == 'traditional') {
            # gel <- gelnet.cv(gdsc_ge[good_indices,], row, 5, 5, nFolds=5, d=rep(1, nrow(penalty)), P=diag(nrow(penalty)))
            gel <- gelnet(gdsc_ge[good_indices,], row, 0.1, 0.5)
        }
        
        # Predict the drug response.
        pred <- tcga_ge %*% gel$w + gel$b
        
        # Add row to the prediction.
        pred_df <- rbind(pred_df, t(pred))
    }
    row.names(pred_df) <- drugs
    write.csv(pred_df, file=paste('./data/gdsc_tcga/dr_matrices/GDSC_TCGA_gelnet_', method_type, '.csv', sep=''))
}

change.files('traditional')