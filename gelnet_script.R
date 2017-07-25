### Author: Edward Huang

library(gelnet)

change.files <- function() {

    gdsc_ge = t(read.table('./data/gdsc_tcga/gdsc_expr_postCB.csv', header=TRUE, row.names=1, sep=',',
    check.names=FALSE))
    gdsc_dr = t(read.table('./data/gdsc_tcga/dr_matrices/gdsc_dr.csv', header=TRUE, row.names=1, sep=',', na.strings = "NA",
    check.names=FALSE))
    
    tcga_ge = t(read.table('./data/gdsc_tcga/tcga_expr_postCB.csv', header=TRUE, row.names=1, sep=',',
    check.names=FALSE))
    tcga_dr = t(read.table('./data/gdsc_tcga/dr_matrices/tcga_dr.csv', header=TRUE, row.names=1, sep=',', na.strings="NA",
    check.names=FALSE))
    
    ppi_mat = read.table('./data/gdsc_tcga/STRING_experimental_KN02_table.csv', header=TRUE, row.names=1, sep=',',
    check.names=FALSE)
    print(dim(ppi_mat))
    penalty = adj2nlapl(ppi_mat)
    
    # Construct the empty prediction matrix.
    pred_df <- data.frame(matrix(ncol = dim(tcga_ge)[1], nrow = 0))
    x <- rownames(tcga_ge)
    colnames(pred_df) <- x
    # y <- rownames(gdsc_dr)
    # row.names(pred_df) <- y
    
    drugs <- colnames(gdsc_dr)
    for(i in 1:ncol(gdsc_dr)) {
        row = gdsc_dr[,i]
        
        good_indices <- which(!is.na(row))
        row <- row[good_indices]
        # print(row)
        gel <- gelnet.cv(gdsc_ge[good_indices,], row, 5, 5, nFolds=5, P = penalty)
        # Predict the drug response.
        pred <- tcga_ge %*% gel$w + gel$b
        
        # Add row to the prediction.
        pred_df <- rbind(pred_df, t(pred))
    }
    row.names(pred_df) <- drugs
    write.csv(pred_df, file='./data/gdsc_tcga/dr_matrices/GDSC_TCGA_gelnet.csv')
    # gelnet.cv(gdsc_ge, gdsc_dr, nFolds = 10)
    # apply(gdsc_dr, 1, gelnet.cv, X=gdsc_ge, nFolds=10)
    # gelnet(gdsc_ge, )
    
    # tcga_row = tcga_dr[,3]
    # tcga_good <- which(!is.na(tcga_row))
    # tcga_row = as.integer(as.factor(tcga_row[tcga_good]))
    # # gelnet.lin.obj(gel$w, gel$b, tcga_ge[tcga_good,], tcga_row, gel$l1, gel$l2)
    # gelnet.lin.obj(gel$w, gel$b, ge_mat[good_indices,], row, gel$l1, gel$l2, P=penalty)
    
    # sum(gel$w $*$ ge_mat[good_indices,]) + gel$b
    # # 882 samples, 13941 genes
    
    # # THIS RIGHT HERE
    # ge_mat[good_indices,] %*% gel$w + gel$b
}

# args <- commandArgs(trailingOnly = TRUE)
# change.files(args[1], args[2])
change.files()