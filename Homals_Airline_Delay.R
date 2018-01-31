.libPaths("/home/admin123/R/x86_64-pc-linux-gnu-library/3.2")
library(dplyr)
library(homals)
library(sampling)
library(gdata)
library(xtable)

set.seed(4314)
setwd("/home/admin123/Clustering_MD/Paper/clustering.experiments/")
fp = "clean_jan_2016_data.csv"
df = read.csv(fp, strip.white = TRUE)
req.cols = setdiff(names(df), c("X", "CANCELLED", "DIVERTED", "FL_NUM"))
df = df[, req.cols]
df = na.omit(df)
df = filter(df, ARR_DELAY <= 155)
xtable(data.frame(t(quantile(df$ARR_DELAY))))
# do the time preprocessing
df = mutate (df,
             CADT  = ifelse ( nchar(CRS_DEP_TIME) < 4,
                              paste(substr(CRS_DEP_TIME,1,1),
                                    substr(CRS_DEP_TIME,2,nchar(CRS_DEP_TIME)), "00", sep = ":"),
                              paste(substr(CRS_DEP_TIME,1,2),
                                    substr(CRS_DEP_TIME,3,nchar(CRS_DEP_TIME)),"00", sep = ":")),
             DDT = paste("2016", "1", df[,"DAY_OF_MONTH"], sep = "-"),
             LADT = paste(DDT, CADT, sep = " "))


df$FDDT = strptime(df$LADT, format =  "%Y-%m-%d %H:%M:%S")
ref.time = strptime("2016-1-1 12:00:00", format =  "%Y-%m-%d %H:%M:%S")
df$NDDT = as.numeric(difftime(df[,"FDDT"], ref.time), units = "mins")

req.cols = setdiff(names(df), c("FDDT", "CADT", "DDT", "LADT", "CRS_DEP_TIME"))
df = df[,req.cols]
df = na.omit(df)
fp.ae = "clean_pp_jan_2016_data.csv"
write.csv(df, fp.ae, row.names = FALSE)
num.cols = c("DEP_DELAY","TAXI_OUT","TAXI_IN", "ARR_DELAY","CRS_ELAPSED_TIME", "NDDT")
df.num = df[num.cols]
fp.num = "num_cols_clean_jan_2016_data.csv"
write.csv(df.num, fp.num, row.names = FALSE)
ord.cols = c("DAY_OF_MONTH", "DAY_OF_WEEK")
nom.cols = c("CARRIER", "ORIGIN", "DEST")
col.in.order = c(ord.cols, nom.cols)
df = df[col.in.order]


#set the day of week and day of month as factors
df$DAY_OF_MONTH = as.factor(df$DAY_OF_MONTH)
df$DAY_OF_WEEK = as.factor(df$DAY_OF_WEEK)

# Run homals on the sample to obtain optimal scaling
var.levels = c( rep("ordinal",2), rep("nominal", 3))

homals.fit = homals(df, ndim = 1, rank = 2, level = var.levels)
df.recoded = as.data.frame(homals.fit$scoremat[,,1])
df.c = cbind(df.recoded, df.num)
df.c = as.data.frame(scale(df.c))
# Write the recoded file out to disk for clustering by mini batch K means
fp = "/home/admin123/Clustering_MD/Paper/clustering.experiments/Jan_2016_Delays_Recoded.csv"
write.csv(df.c, fp, row.names = FALSE)

