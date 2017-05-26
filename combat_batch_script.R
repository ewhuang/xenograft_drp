
# Gene expression: edata
# 22283x57 table:
#         GSM71019.CEL    GSM71020.CEL
# 1007_s_at   10.115170   8.628044
# 1053_at 5.345168    5.063598


# pheno:
#                 sample  outcome batch   cancer
# GSM71019.CEL    1       Normal  3       Normal
# GSM71020.CEL    2       Normal  2       Normal

library(devtools)
library(Biobase)
library(sva)
library(bladderbatch)
library(snpStats)

data(bladderdata)
pheno = pData(bladderEset)
edata = exprs(bladderEset)

batch=pheno$batch
modcombat = model.matrix(~1, data=pheno)
modcancer = model.matrix(~cancer, data=pheno)
combat_edata = ComBat(dat=edata, batch=batch, mod=modcombat, par.prior=TRUE, prior.plots=FALSE)