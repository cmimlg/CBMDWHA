.libPaths("/home/admin123/R/x86_64-pc-linux-gnu-library/3.2")
library(dplyr)
library(homals)
library(sampling)
library(gdata)

set.seed(4314)
setwd("/home/admin123/Clustering_MD/Paper/clustering.experiments/")
fp = "Syn_Mixed_Data.csv"
df = read.csv(fp, strip.white = TRUE)
df = na.omit(df)
df.num = df[c("N1", "N2")]
df.clus = df["C"]
# don't need the or numerical columns or Cluster column for homals analysis
req.cols = setdiff(names(df), c("C", "N1", "N2"))
df = df[req.cols]
# Run homals on the sample to obtain optimal scaling
var.levels = c( rep("nominal", 2))

homals.fit = homals(df, ndim = 1, rank = 2, level = var.levels)
df.recoded = as.data.frame(homals.fit$scoremat[,,1])
df.c = cbind(df.recoded, df.num)
df.c = as.data.frame(scale(df.c))
df.c = cbind(df.c, df.clus)
# Write the recoded file out to disk for clustering analysis
fp = "/home/admin123/Clustering_MD/Paper/clustering.experiments/Synthetic_Data_Recoded.csv"
write.csv(df.c, fp, row.names = FALSE)

