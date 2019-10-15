#######################################
#   Purpose : Market Basket Analysis
#   Author  : Gana Aruneswaran
#   Date    : 10/10/2019
#######################################

## Import Packages

#install.packages("assertr")
#install.packages("arulesViz")  
#install.packages("arules") 
library("assertr")
library("arules")
library("arulesViz")

#read in initial dataset

MA_Start <- read.table("C:\\Users\\GA088632\\Documents\\Trial_MA_P4.txt"
                        , sep="\t", header=TRUE,stringsAsFactors = F
                        , na.strings = c("?",NULL))
# get rid of strange ids (i.e. -1)

MA_Start <- MA_Start[MA_Start$Ord_Id > 0,]

# save ord_ids

x = unique(MA_Start$Ord_Id)

# initialise df to keep order details

df_ma_init = data.frame()

# create a loop to insert records as appropriate

for (val in x){
  ante_tr <- MA_Start[MA_Start$Ord_Id == val,]
  ante_tr[paste("",val,"")]= ante_tr["Ord_Id"]
  post_tr <- t(ante_tr)
  next_tr <- t(post_tr[2,])
  one_cell <- as.data.frame(col_concat(next_tr, sep = ","))
  colnames(one_cell) = "Products"
  one_cell["Ord_Id"] <- val
  one_cell <- one_cell[,c(2,1)]
  df_ma_init <- rbind(df_ma_init,one_cell)
}

# Write up csv that has products in format 

write.csv(df_ma_init[,2],"C:\\Users\\GA088632\\Documents\\market_basket_transactions.csv", quote = FALSE, row.names = FALSE)

# read back in as transactional dataset

tr <- read.transactions('C:\\Users\\GA088632\\Documents\\market_basket_transactions.csv', format = 'basket', sep=',')

# quick set of summary statistics

summary(tr)

if (!require("RColorBrewer")) {
  # install color package of R
  install.packages("RColorBrewer")
  #include library RColorBrewer
  library(RColorBrewer)
}
itemFrequencyPlot(tr,topN=20,type="absolute",col=brewer.pal(8,'Pastel2'), main="Absolute Item Frequency Plot")
# can change topN or type = "relative"

# Use apriori algorithm

association.rules <- apriori(tr, parameter = list(supp=0.0001, conf=0.5,maxlen=10))
summary(association.rules)
inspect(association.rules[1:5])

# can remove rules if chosen to (haven't in this case)

subset.rules <- which(colSums(is.subset(association.rules, association.rules)) > 1) # get subset rules in vector
length(subset.rules)  #> 3913
subset.association.rules. <- association.rules#[-subset.rules] # remove subset rules.

# inspect rules
inspect(head(subset.association.rules.))

#grpah rules
subRules<-association.rules[quality(association.rules)$confidence>0.4]
#Plot SubRules
plot(subRules)
plot(subRules,method="two-key plot")

# show this using a flow chart/quasi-Markov graph
top10subRules <- head(subRules, n = 6, by = "confidence")
plot(top10subRules, method = "graph",  engine = "htmlwidget")

# Filter top 20 rules with highest lift
subRules2<-head(subRules, n=10, by="lift")
plot(subRules2, method="paracoord")