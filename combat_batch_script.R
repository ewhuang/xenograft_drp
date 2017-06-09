# Basic template for running ComBat.

# Variable edata: gene expression file
# gene x sample:
#               GSM71019.CEL    GSM71020.CEL
# 1007_s_at     10.115170       8.628044
# 1053_at       5.345168        5.063598

# Variable pheno: sample description file.
#                 sample  outcome batch   cancer
# GSM71019.CEL    1       Normal  3       Normal
# GSM71020.CEL    2       Normal  2       Normal

# library(devtools)
# library(Biobase)
library(sva)
# library(snpStats)

# change.files <- function(expr_fname, pheno_fname){
change.files <- function(drug){
    # data(bladderdata)
    # edata = exprs(bladderEset)
    # pheno = pData(bladderEset)
    edata = read.csv(paste('./data/', drug, '_gene_expr_before_combat.csv', sep=''))
    edata2 <- edata[,-1]
    rownames(edata2)<-edata[,1]
    pheno = read.table(file=paste('./data/', drug, '_pheno.txt', sep=''),
        sep = '\t', header = TRUE)
    pheno2 <- pheno[,-1]
    rownames(pheno2)<-pheno[,1]

    batch=pheno2$batch
    modcombat = model.matrix(~1, data=pheno2)
    combat_edata = ComBat(dat=edata2, batch=batch, mod=modcombat, par.prior=TRUE,
        prior.plots=FALSE)
    write.csv(combat_edata, file=paste('./data/', drug,
        '_gene_expr_after_combat.csv', sep=''))
}

args <- commandArgs(trailingOnly = TRUE)
for (arg in args) {
    change.files(arg)
}