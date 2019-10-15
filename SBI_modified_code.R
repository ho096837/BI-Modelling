##load packages
install.packages(c("zoo", "ggpubr", "RODBC","randomForest", 
                   "e1071", "rpart","caret", "corrplot", "Hmisc",
                   "ROSE", "corrplot", "tidyverse", "Amelia", "Boruta", 
                   "mlbench", "RANN", "xgboost", "lubridate", "survminer", 
                   "xgboost", "DataExplorer", "skimr", "pirate", "yarrr", "ggfortify",
                   "survminer", "mlr", "factorMerger", "ingredients"))



suppressPackageStartupMessages(library(skimr))
library(factorMerger)
library(ingredients)
library(mlr)
library(ggfortify)
library(survminer)
library(DataExplorer)
library(RODBC)
library(plyr)
#library(dplyr)
library(lubridate)
#library(glmulti)
#library(data.table)
library(Hmisc)
library(randomForest)
library(rpart)
library(ggplot2)
library(e1071)
library(caret)
library(corrplot)
library(stringr)
library(ROSE)
library(tidyverse)
library(Amelia)
library(Boruta)
library(missForest)
library(lattice)
library(mlbench)
library(RANN)
library(h2o)
library(xgboost)
library(ggpubr)
library(lubridate)
library(zoo)
library(yarrr)
library(pirate)

#set working directory
getwd()
setwd("C:/Users/ho096837/Desktop/datasets")


#read in relevant files
desc_file <- read.table("eddemm_file.txt", 
                        sep = "\t", 
                        header = TRUE, 
                        fill = TRUE,
                        comment.char = "#",  
                        na.strings = c("?", NULL) ,
                        quote = "", 
                        stringsAsFactors = TRUE)

Life_Cycles <- read.table("Life_Cycles.txt", 
                          sep = "\t", 
                          header = TRUE, 
                          fill = TRUE,
                          comment.char = "#",  
                          na.strings = c("?", NULL) ,
                          quote = "", 
                          stringsAsFactors = TRUE)



Age_data <- read.table("Age_data.txt", 
                       sep = "\t", 
                       header = TRUE, 
                       fill = TRUE,
                       comment.char = "#",  
                       na.strings = c("?", NULL) ,
                       quote = "", 
                       stringsAsFactors = TRUE)

device <- read.csv("device.csv", 
                   sep = ",", 
                   header = TRUE,
                   stringsAsFactors = TRUE)


Upgrade_RPC <- read.table("Upgrade_RPC.txt", 
                          sep = "\t", 
                          header = TRUE, 
                          fill = TRUE,
                          comment.char = "#",  
                          na.strings = c("?", NULL) ,
                          quote = "", 
                          stringsAsFactors = TRUE)



str(Upgrade_RPC)


##QUICK DATA EXPLORATION
str(desc_file)
create_report(desc_file)
create_report(desc_file, y = "Event")
#introduce(desc_file)
plot_intro(desc_file)
plot_missing(desc_file)
plot_bar(desc_file)
plot_histogram(desc_file)
plot_correlation(desc_file)
plot_qq(desc_file)
plot_qq(desc_file, by = "Event")
plot_boxplot(desc_file)
plot_boxplot(desc_file, by = "Event")


skim(desc_file) %>% skimr::kable()

#start wrangling data
str(desc_file)

#rename to common column SUBS_ID
Age_data <- Age_data %>%
  dplyr::rename(SUBS_ID = Subscription.ID)

#change some data types
Age_data$SUBS_ID = as.character(Age_data$SUBS_ID)
desc_file$subscr_id <- as.character(desc_file$subscr_id)
Life_Cycles$SUBS_ID <- as.character(Life_Cycles$SUBS_ID)
device$Dvc_Prod_Cd <- as.factor(device$Dvc_Prod_Cd)
Upgrade_RPC$Subs_Id <-  as.character(Upgrade_RPC$Subs_Id)

#select relevant columns from Upgrade_RPC data
Upgrade_RPC <- Upgrade_RPC %>%
  select(Subs_Id, Upgrade_Previous_FV, Upgrade_Current_FV, upgrade_delta, 
         RPC_Without_Upgrades_Previous_FV, RPC_Without_Upgrades_Current_FV,RPC_delta, No_of_Services) %>%
 dplyr::rename(SUBS_ID = Subs_Id) %>%
  filter(SUBS_ID != -1)

#change multiple columns to date format
for (i in c(2,3,4,12,15,16,18,19)){
  desc_file[,i] <- as.Date(as.character(desc_file[,i]), format = "%d/%m/%Y" )
}

for (i in c(9,10,13,21)){
  desc_file[,i] <- as.factor(desc_file[,i])
}

str(desc_file)




#more wrangling and feature engineering

desc_file <- desc_file  %>%
  select(-Subs_Id, -subss_id, -PROJECT, -Conng_Actvn_Chnl, -PrtOUT_Dt ) %>%
  mutate(conn_month = lubridate::month(Conn_Dt, label = T, abbr = T), 
         dsconn_year = lubridate::year(Dsconn_Dt),  
         dsconn_month = lubridate::month(Dsconn_Dt, label = T, abbr= T),  
         tenure = round(Conn_Dt %--% Dsconn_Dt/dweeks(1), 0), 
         dscdt = as.Date(replace_na((Dsconn_Dt = "01/01/2030")), format = "%d/%m/%Y"))  %>%
  unite(dsconn_ym, c("dsconn_year","dsconn_month"), sep = "-")%>%
  mutate(face_val = as.factor(case_when(Face_Value == "22" ~ "22", 
                                        Face_Value == "40" ~ "40",
                                        Face_Value == "50" ~ "50",
                                        Face_Value == "60" ~ "60",
                                        Face_Value == "70" ~ "70",
                                        Face_Value == "80" ~ "80",
                                        Face_Value == "100" ~ "100",
                                        TRUE ~ "other")))%>%
  mutate(status = case_when(is.na(Dsconn_Dt) ~ "active", TRUE ~ "inactive"))%>%
  mutate(day_of_year = lubridate::yday(Conn_Dt)) %>%
  mutate(week_of_year = lubridate::week(Conn_Dt)) %>%
  #mutate(doy_mpp = lubridate::yday(Mobile.Payment.Plan.Contract.Start.Date)) %>%
  # mutate(woy_mpp = lubridate::week(Mobile.Payment.Plan.Contract.Start.Date)) %>%
  mutate(doy_cpl = lubridate::yday(Plan.Contract.Start.Date)) %>%
  mutate(woy_cpl = lubridate::week(Plan.Contract.Start.Date)) %>%
  #mutate(mpp_month = lubridate::month(Mobile.Payment.Plan.Contract.Start.Date, label = TRUE, abbr = TRUE))%>%
  mutate(cpl_month = lubridate::month(Plan.Contract.Start.Date, label = TRUE, abbr = TRUE)) %>%
  mutate(port_telstra_Optus = case_when(PrtOUT_SP_Cd == "10057" ~ "Telstra",
                                        PrtOUT_SP_Cd == "10217"  ~ "Telstra",
                                        PrtOUT_SP_Cd == "10000" ~ "Optus",
                                        PrtOUT_SP_Cd == "10737" ~ "Optus",
                                        PrtOUT_SP_Cd == "10708" ~ "IspONE",
                                        PrtOUT_SP_Cd == "10136" ~ "Amasysm",
                                        PrtOUT_SP_Cd == "10724" ~ "Kogan",
                                        TRUE ~ "Other")) %>%
  mutate(dmaf_bins = case_when(DMAF <31 ~ "<=30",
                               DMAF <= 40 ~ "31-40",
                               DMAF <= 50 ~ "41-50",
                               DMAF <= 60 ~ "51-60",
                               TRUE ~ "60 and above")) %>%
  mutate(dmaf_bins_2 = case_when(DMAF <31 ~ "<=30",
                                 DMAF <= 40 ~ "31-40",
                                 DMAF <= 45 ~ "41-45",
                                 DMAF <= 50 ~ "46-50",
                                 TRUE ~ "50 and above")) %>%
  dplyr::rename(SUBS_ID = subscr_id)  %>%
  left_join(Life_Cycles, by = "SUBS_ID") %>%
  select(-PRD_DT, -rownum) %>%
  left_join(Age_data, by = "SUBS_ID")  %>%
  dplyr::rename(Dvc_Prod_Cd = Provisioned.Device.Product.Code)  %>%
  left_join(device, by = "Dvc_Prod_Cd") %>%   
  mutate(SIMO_vs_nonSIMO = case_when(str_sub(Contract_Group, -4,-1) == "SIMO" ~ "SIMO",
                                     TRUE ~ "NON_SIMO")) %>%
  mutate(Contract_vs_non_contract = case_when(substring(Contract_Group, 1, 2)  == "No" ~ "No_Contract",
                                              TRUE ~ "Contract")) %>%
  mutate(mpp_yes_no = case_when(is.na(Earliest.Mobile.Payment.Plan.Contract.Start.Date) ~ "No",
                                TRUE ~ "Yes")) %>%
  mutate(Plan_contract = case_when(is.na(Plan.Contract.Start.Date) ~ "No", 
                                   TRUE ~ "Yes")) %>%
  mutate(contract_yes_no = case_when(Contract_vs_non_contract == "Contract" ~ "yes",
                                     Contract_vs_non_contract == "No_Contract" ~ "no",
                                     mpp_yes_no == "Yes" ~ "yes",
                                     mpp_yes_no == "No" ~ "no")) %>%
  mutate(event_group = case_when(Event == "Disconnect" ~ "Disconnect",
                                 Event == "Port Out" ~ "Disconnect",
                                 Event == "Upgrade" ~ "Upgrade",
                                 TRUE ~ "AS-IS"))   %>%
  left_join(Upgrade_RPC, by = "SUBS_ID") %>%
  mutate(upgrade_downgrage = case_when(upgrade_delta < 0 ~ "Downgrade",
                                       upgrade_delta > 0 ~ "upgraded", 
                                       TRUE ~ "unchanged")) %>%
  mutate(num_ser = case_when(No_of_Services == 1 ~ "1",
                             No_of_Services == 2 ~ "2",
                             No_of_Services == 3 ~ "3",
                             No_of_Services == 4 ~ "4",
                             TRUE ~ "5+"))



#investigate with pirate plots
#pirateplot(formula = tenure ~ Contract_Length, 
#           data = desc_file,
#           main = "Pirateplot of DMAF by tenure",
#           xlab = "Contract_length",
#           ylab = "tenure")


View(colnames(desc_file))


#invetsigating tenures before disconnection
tenure_plot <- function (df = desc_file, x, y) {
  tenure_boxplot <- df %>%
    filter(status == "inactive") %>%
    ggplot(aes_string(x, y)) + geom_boxplot(aes(color = x)) 
  tenure_violin_plot <-   df %>%
    filter(status == "inactive") %>%
    ggplot(aes_string(x,  y)) + geom_violin(trim = F, aes(fill = x)) 
  print(tenure_boxplot)
  print(tenure_violin_plot)
}

tenure_plot(desc_file, "face_val", "tenure")



cat_features <- c("event_group", "Contract_Length", "face_val", "SIMO_vs_nonSIMO",
                  "Contract_vs_non_contract", "mpp_yes_no", "contract_yes_no", 'num_ser',
                  "dmaf_bins", "dmaf_bins_2" )



for (i in cat_features ) {
  tenure_plot <- function (df = desc_file, x, y = tenure) {
    tenure_boxplot <- df %>%
      filter(status == "inactive") %>%
      ggplot(aes_string(x, y)) + geom_boxplot(aes(color = x)) 
    tenure_violin_plot <-   df %>%
      filter(status == "inactive") %>%
      ggplot(aes_string(x,  y)) + geom_violin(trim = F, aes(fill = x)) 
    print(tenure_boxplot)
    print(tenure_violin_plot)
  }
  
  tenure_plot(desc_file, i, "tenure")
  
}



tenure_plot_hist <- function (df = desc_file, x) {
  tenure_hist <- df %>%
    filter(status == "inactive") %>%
    ggplot(aes_string(x)) + geom_histogram(bins = 200, fill = "blue") +
    facet_wrap(vars(x))
  print(tenure_hist)
}

tenure_plot_hist(event_group)


plt_tenure_dscnn <- desc_file %>%
  filter(status == "inactive") %>%
  ggplot(aes(tenure)) + geom_histogram(bins = 200, fill = "blue") + ggtitle('Histogram of Disconnection')
plt_tenure_dscnn


tenure_hist <- desc_file %>%
  filter(status == "inactive") %>%
  ggplot(aes_string("tenure")) + geom_histogram(bins = 200, fill = "blue") +
  facet_wrap(vars(face_val))



### Most of the descriptives are going to be around visualising tenures before disconnections to generate theories around risk elements
#Chart Intro
texta = paste("\n   We are following 245,000 odd customers.\n",
              "       Who are they \n",
              "           how did we recruit them")
ggplot() + 
  annotate("text", x = 4, y = 25, size=8, label = texta) + 
  theme_bw() +
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank())  + 
  theme(axis.title = element_blank()) + 
  theme(axis.text.x=element_blank()) + 
  theme(axis.text.y=element_blank())


##AGE
plt_age_band <- desc_file %>%
  ggplot(aes(AGE_BAND_NUM)) + geom_bar(aes(fill = AGE_BAND_NUM)) +
  ggtitle('Age Bands') +  theme(legend.position = "none")



plt_age_hist <- desc_file %>%
  ggplot(aes(AGE_NUM)) + geom_histogram(bins = 200, fill = "cyan") +
  ggtitle('Histogram of Age') 


plt_age_hist_CL <- desc_file %>%
  ggplot(aes(AGE_NUM)) + geom_histogram(bins = 200, fill = "cyan") + facet_wrap(vars(Contract_Length)) +
  ggtitle('Histogram of Age by Contract Length')


ggarrange(plt_age_band, plt_age_hist, 
          plt_age_hist_CL, ncol = 2, 
          nrow = 2, common.legend = F)


##FACE VALUE
plt_FV_overall <- ggplot(desc_FV, aes(x = face_value , y = perc_FV, fill = face_value)) + 
  geom_bar(stat = "identity") + 
  theme(legend.position = "none")# +scale_x_discrete(limits=c("25","30","40"))



#Contract Length
plt_ctr_len_overall <- ggplot(desc_contract, aes(x = Contract_Length , y = perc_contract, fill = Contract_Length)) + 
  geom_bar(stat = "identity") + theme(legend.position = "none")

#face value
ggarrange(plt_FV_overall, 
          plt_ctr_len_overall ,
          ncol = 2, nrow = 2, common.legend = F)



text = paste("\n   We are following 245,000 customers.\n",
             "       Now let's look at how we recruited them \n",
             "           across channel, bands and other dimensions")
ggplot() + 
  annotate("text", x = 4, y = 25, size=8, label = text) + 
  theme_bw() +
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank())  + 
  theme(axis.title = element_blank()) + 
  theme(axis.text.x=element_blank()) + 
  theme(axis.text.y=element_blank())




ggarrange(plt_conn_mnth, plt_conn_mnth_CL, 
          plt_conn_mnth_fv, 
          ncol = 2, nrow = 2, common.legend = F)



##EVENTS 
text1 = paste("\n   What happened to them along their journey?.\n",
              "      did they stay or leave us? \n",
              "           let's find out")
ggplot() + 
  annotate("text", x = 4, y = 25, size=8, label = text1) + 
  theme_bw() +
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank())  + 
  theme(axis.title = element_blank()) + 
  theme(axis.text.x=element_blank()) + 
  theme(axis.text.y=element_blank())

plt_event_overall <- ggplot(desc_event, aes(x = event_group, y = perc_event,fill = event_group)) + 
  geom_bar(stat = "identity") + theme(legend.position = "none") +
  ggtitle("Events")   + 
  theme(legend.title = element_blank())


#EVENT BY CL
plt_event_contract_length <- desc_file %>%
  ggplot(aes(Contract_Length, fill = event_group)) + geom_bar(position = "fill") + 
  ggtitle('Event by Contract Length') + theme(legend.title = element_blank())

plt_event_Age_Band <- desc_file %>%
  ggplot(aes(AGE_BAND_NUM, fill = event_group)) + geom_bar(position = "fill") + 
  ggtitle('Event by Age Band') + theme(legend.title = element_blank())

##EVENT BY FV
plt_event_FV <- desc_file %>%
  ggplot(aes(face_val, fill = event_group)) + geom_bar(position = "fill") + ggtitle('Event by face value') + 
  theme(legend.title = element_blank())




ggarrange(plt_event_overall, plt_event_contract_length, 
          plt_event_FV, plt_event_Age_Band,
          ncol = 3, nrow = 2, common.legend = F)



#DISCONNECTIONS 

text2 = paste("\n   Now we know what happened to them!.\n",
              "      Now we can investigate disconnections further. \n",
              "           how did they disconnect?")
ggplot() + 
  annotate("text", x = 4, y = 25, size=8, label = text2) + 
  theme_bw() +
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank())  + 
  theme(axis.title = element_blank()) + 
  theme(axis.text.x=element_blank()) + 
  theme(axis.text.y=element_blank())




plt_dscnt_FV <- ggplot(desc_file, aes(x = Dsconn_Dt)) + 
  stat_count(aes(fill = face_val), position = "fill") + 
  theme(legend.position = "right") +
  ggtitle(" Monthly Disconnections by Face Value ") +
  theme(legend.title = element_blank())

plt_dscnt_FV



#invetsigate patterns arund Telstra and Optus disocnnections 
bgt <- desc_file %>%
  group_by(Dsconn_Dt  ) %>% 
  count(port_telstra_Optus) %>%
  filter(port_telstra_Optus != "Other")

Telstra_Optus_discnn <- ggplot(bgt, aes(x = Dsconn_Dt, y = n))+  
  geom_line(aes(fill = port_telstra_Optus, color = port_telstra_Optus ))  + 
  coord_cartesian( ylim = c(0, 1500))
Telstra_Optus_discnn




#Correlations

text5 = paste("\n   What factors are somehow related to how long\n",
              "      these customers stay with us? \n",
              "           let's find out")
ggplot() + 
  annotate("text", x = 4, y = 25, size=8, label = text5) + 
  theme_bw() +
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank())  + 
  theme(axis.title = element_blank()) + 
  theme(axis.text.x=element_blank()) + 
  theme(axis.text.y=element_blank())


num_data <-  split_columns(desc_file)$continuous

corrlnts <- round(cor(num_data, use = "pairwise.complete.obs", method = "pearson"),2)

corrplot::corrplot(corrlnts, method = "number", type = "lower", main = "\n\n Correlation Plots")

#core_data_binarize <- desc_file %>%
#  select_if(negate(is.Date)) %>%
# select(-c(Dsconn_Dt, Prd_Dt, AGE_NUM, PROJECT, dsconn_ym, Face_Value, tenure, timetenure))

#core_data_binarize$Mobile.Payment.Plan.Contract.Term <- as.numeric(core_data_binarize$Mobile.Payment.Plan.Contract.Term)
#core_data_binarize$Plan.Contract.Term <- as.numeric(core_data_binarize$Plan.Contract.Term)



#binarized_data <- binarize(core_data_binarize, n_bins = 4, thresh_infreq = 0.01,
# name_infreq = "-OTHER", one_hot = TRUE)

#rate plan changes
plt_upgrdowngr <- desc_file %>%
  ggplot(aes(upgrade_downgrage, upgrade_delta)) + geom_boxplot(aes(color = upgrade_downgrage))  + ggtitle("Box Plot of Upgrade/Downgrade")
plt_upgrdowngr
qplot(y = desc_file$upgrade_delta, x= 1, geom = "boxplot")
mean(desc_file$upgrade_delta, na.rm = T)


plt_tnr_delat_upgr <- desc_file %>%
  ggplot(aes(upgrade_downgrage, upgrade_delta)) + geom_boxplot(aes(color = upgrade_downgrage))  + ggtitle("Actual Tenures ")
plt_tnr_delat_upgr



desc_file__nna <- desc_file %>%
  mutate(upgrade_bands =   cut(upgrade_delta, breaks = c( -100,-50, -20, 0, 20, 50, 70, Inf), 
                               labels = c("[-100, -50]", "[-49, -20]",
                                          "[-19 0]", "[1, 20]", "[21, 50]",
                                          "[51, 70]", "[71, 100]") )) %>%
  filter(!is.na(upgrade_bands)) %>%
  mutate(survived = (case_when(status == "active" ~ 0,
                               status == "inactive" ~ 1)))

desc_file__nna$timetenure <- ifelse(is.na(desc_file__nna$tenure), 
                                    round(desc_file__nna$Conn_Dt %--% today()/dweeks(1), 0), 
                                    desc_file__nna$tenure)

unique(desc_file__nna$upgrade_bands)



##SURVIVAL ANALYSIS

text6 = paste("\n   What is the probability for a customer recruited today?")

ggplot() + 
  annotate("text", x = 4, y = 25, size=8, label = text6) + 
  theme_bw() +
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank())  + 
  theme(axis.title = element_blank()) + 
  theme(axis.text.x=element_blank()) + 
  theme(axis.text.y=element_blank())


desc_file <-  desc_file %>%
  mutate(life_time = round(Conn_Dt %--% dscdt/dweeks(1), 
                           0)) %>%
  mutate(survived = (case_when(status == "active" ~ 0,
                               status == "inactive" ~ 1))) %>%
  mutate(tnr = replace_na(tenure, 1000)) %>%
  filter(tnr >= 0, Contract_Length != "36") %>%
  mutate(contr_leng_band = case_when(Contract_Length == "0" ~ 1,
                                     Contract_Length == "12" ~ 2,
                                     Contract_Length == "24" ~ 3))


#filter out NAs from tenure

desc_file$timetenure <- ifelse(is.na(desc_file$tenure), 
                               round(desc_file$Conn_Dt %--% today()/dweeks(1), 0), 
                               desc_file$tenure)



str(desc_file)

#create a survival object
surv.object <- with(desc_file, Surv(timetenure, 
                                    survived))
#fit the survival model to the object
surv.fit <- survfit(Surv(timetenure, survived) ~ 1, 
                    data = desc_file)
#check survival times between 0 and 180 weeks
surv.fit.summ <- summary(surv.fit, 
                         times = c(0,12, 24, 36,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128))

#autoplot(surv.fit, xlim = c(0,150))
ggsurvplot(surv.fit,
           data = desc_file,
           surv.median.line = "hv",
           xlab = "Time in weeks",  
           break.time.by = 20,        
           ggtheme = theme_bw(),
           xlim = c(0,150),
           ylim = c(0.5, 1),
           title = "overall survival probabilities" )


#fit the survival model across Contract length
surv.plan.fit <- survfit(Surv(timetenure, survived) ~ contr_leng_band, 
                         data = desc_file)
surv.fit.plan.summ <-  summary(surv.plan.fit, 
                               times = c(0,12, 24, 36,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128))

#autoplot(surv.plan.fit, 
#xlim = c(0,180))


ggsurvplot(
  surv.plan.fit ,         
  data = desc_file,      
  risk.table = FALSE,
  ncensor.plot = FALSE,
  title = " probability of survival by contract length",
  pval = TRUE,          
  conf.int = F, 
  surv.median.line = "hv",
  fun = "pct",
  linetype = "strata",
  xlim = c(0,150),       
  xlab = "Time in weeks",  
  break.time.by = 20,        
  ggtheme = theme_bw(),   
  #risk.table.y.text.col = "strata", # colour risk table text annotations.
  #risk.table.y.text = FALSE,
  legend = "bottom",
  legend.title = "Contract Length",
  legend.labs = c("0",
                  "12",
                  "24")# show bars instead of names in text annotation in legend of risk table
)

#plot out event and hazard rates as well
ggsurvplot(surv.plan.fit, 
           data = desc_file, 
           fun = "event", xlim = c(0,150))
ggsurvplot(surv.plan.fit, 
           data = desc_file, 
           fun = "cumhaz", 
           xlim = c(0,150))


df_surv_plan <- data.frame(time = surv.plan.fit$time,
                           n.risk = surv.plan.fit$n.risk,
                           n.event = surv.plan.fit$n.event,
                           n.censor = surv.plan.fit$n.censor,
                           surv = surv.plan.fit$surv,
                           upper = surv.plan.fit$upper,
                           lower = surv.plan.fit$lower
)

df_surv_plan

summary(surv.plan.fit)$table


#surviva by face value
surv.fv.fit <- survfit(Surv(timetenure, survived) ~ face_val, 
                       data = desc_file)
surv.fit.fv.summ <-  summary(surv.fv.fit, 
                             times = c(0,12, 24, 36,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128))

#autoplot(surv.fv.fit, xlim = c(0,180))

ggsurvplot(
  surv.fv.fit ,       
  data = desc_file,      
  risk.table = F,
  ncensor.plot = F,
  pval = TRUE,          
  conf.int = F, 
  surv.median.line = "hv",
  title = "probability of survial by face value",
  fun = "pct",
  linetype = "strata",
  xlim = c(0,150),       
  xlab = "Time in weeks",  
  break.time.by = 20,        
  ggtheme = theme_bw(),   
  #risk.table.y.text.col = "strata", # colour risk table text annotations.
  #risk.table.y.text = FALSE,
  legend = "bottom",
  legend.title = "face value",
  legend.labs = c("25",
                  "30",
                  "40",
                  "50",
                  "60",
                  "70",
                  "80",
                  "other"))






#generate interaction plots across multpiple variabales
survfit(Surv(timetenure, survived) ~ face_val, 
        data = desc_file)

surv.interaction.fit <- survfit(Surv(timetenure, survived) ~ face_val + Contract_Length, 
                                data = desc_file)

ggsurv <- ggsurvplot(surv.interaction.fit, fun = "pct", conf.int = FALSE,
                     ggtheme = theme_bw(), surv.median.line = "hv")

ggsurv$plot + theme_bw() + 
  theme (legend.position = "right")+
  ggtitle("face value faceted by contract length")+
  facet_grid(vars(Contract_Length))

#significance check
surv_diff_check <- survdiff(Surv(timetenure, survived) ~ face_val, 
                            data = desc_file)
surv_diff_check





# Fit a Cox proportional hazards model
desc_file_cox <- desc_file %>%
  mutate(upgrade_bands =   cut(upgrade_delta, breaks = c( -100,-50, -20, 0, 20, 50, 70, Inf), 
                               labels = c("[-100, -50]", "[-49, -20]",
                                          "[-19 0]", "[1, 20]", "[21, 50]",
                                          "[51, 70]", "[71, 100]") ))

str(desc_file_cox)

set.seed(123)
surv.object_cox <- with(desc_file_cox, Surv(timetenure, 
                                            survived))



fit.coxph<- coxph(surv.object_cox ~ face_val + SIMO_vs_nonSIMO + contract_yes_no + mpp_yes_no + upgrade_bands + upgrade_downgrage + dmaf_bins_2+ 
                    num_ser+ dmaf_bins, data = desc_file_cox)

summary_cpxph <- summary(fit.coxph)

ggforest(fit.coxph, data = desc_file_cox)
coxph_dataframe <- data.frame(summary_cpxph$conf.int, summary_cpxph$coefficients, summary_cpxph$n, summary_cpxph$nevent)
coxph_dataframe <- cbind(Attribute =rownames(coxph_dataframe), coxph_dataframe)
coxph_dataframe <- coxph_dataframe %>%
  mutate(hazard_factor = 1 - exp.coef.)
head(coxph_dataframe)

write.csv(coxph_dataframe, "cox_results_data")

#predict(fit,newdata = data_test,   type="expected")


#ftest <-  cox.zph(fit.coxph)
#ftest
#ggcoxzph(ftest)


##SEGMENTATION FOR RISK ZONES
str(desc_file)

desc_file_m <- desc_file %>%
  mutate(tzone = as.factor(case_when (Contract_Length == "0" & status == "active" & timetenure <=12 ~ "0_rz_1_active",
                                      Contract_Length == "0" & status == "inactive" & timetenure <=12 ~ "0_rz_1_inactive",
                                      Contract_Length == "0" & status == "active" & timetenure >12 ~ "0_post_rz_1_active",
                                      Contract_Length == "0" & status == "inactive" & timetenure >12 ~ "0_post_rz_1_inactive",
                                      Contract_Length == "12" & status == "active" & timetenure <=12 ~ "12_rz_1_active",
                                      Contract_Length == "12" & status == "inactive" & timetenure <=12 ~ "12_rz_1_inactive",
                                      Contract_Length == "12" & status ==  "inactive" & timetenure > 12 &  timetenure <= 46 ~"12_post_rz_1_inactive",
                                      Contract_Length == "12" & status == "active" & timetenure > 12 & timetenure <= 46 ~ "12_post_rz_1_active",
                                      Contract_Length == "12" & status == "active" & timetenure > 46 & timetenure <= 56 ~ "12_rz_2_active",
                                      Contract_Length == "12" & status == "inactive" & timetenure > 46 & timetenure <= 56 ~ "12_rz_2_inactive",
                                      Contract_Length == "12" & status == "active" & timetenure > 56 ~ "12_post_rz_2_active",
                                      Contract_Length == "12" & status == "inactive" & timetenure > 56 ~ "12_post_rz_2_inactive",
                                      Contract_Length == "24" & status == "active" & timetenure <=12 ~ "24_rz_1_active",
                                      Contract_Length == "24" & status == "inactive" & timetenure <=12 ~ "24_rz_1_inactive",
                                      Contract_Length == "24" & status == "active" & timetenure > 12 & timetenure <=98 ~ "24_post_rz_1_active",
                                      Contract_Length == "24" & status == "inactive" & timetenure > 12 & timetenure <=98 ~ "24_post_rz_1_inactive",
                                      Contract_Length == "24" & status == "active" & timetenure > 98 & timetenure <= 108 ~ "24_rz_2_active",
                                      Contract_Length == "24" & status == "inactive" & timetenure > 98 & timetenure <= 108 ~ "24_rz_2_inactive",
                                      Contract_Length == "24" & status == "inactive" & timetenure > 108 ~ "24_post_rz_2_inactive",
                                      Contract_Length == "24" & status == "active" & timetenure > 108 ~"24_post_rz_2_active",
                                      TRUE ~ "other"
  )))




text7 = paste("\n   Now we can develop segments out of the risk factors we saw earlier.\n",
              "      are there factors that sort of predispose a customer towards existing \n",
              "           in a risk zone?")
ggplot() + 
  annotate("text", x = 4, y = 25, size=8, label = text7) + 
  theme_bw() +
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank()) + 
  theme(axis.title = element_blank()) + 
  theme(axis.text.x=element_blank()) + 
  theme(axis.text.y=element_blank())



plt_rzones <- ggplot(desc_file_m, aes(x = tzone)) + 
  stat_count(aes(fill = tzone)) +
  theme(legend.position = "NULL") + 
  ggtitle(" Time Zones/ Risk Zones") + coord_flip()
plt_rzones





#MODELING
str(desc_file)
#drop unnecessary variables
desc_file_model <-  desc_file %>%
  select(-c(Conn_Dt, Dsconn_Dt, Event,
            Face_Value, Prd_Dt, PrtOUT_SP_Cd, Plan.Contract.Start.Date,
            Plan.Contract.End.Date, Earliest.Mobile.Payment.Plan.Contract.Start.Date, 
            Latest.Mobile.Payment.Plan.Contract.End.Date,  Dvc_Prod_Cd, 
            dsconn_ym, tenure, dscdt, status, port_telstra_Optus,Contract_Group,
            AGE_NUM, Mnfctr_Nm, Dvc_Gnrtn,Upgrade_Previous_FV, Upgrade_Current_FV, upgrade_delta,
            RPC_Without_Upgrades_Previous_FV, RPC_Without_Upgrades_Current_FV, 
            RPC_delta,survived, tnr, contr_leng_band,
            timetenure, life_time, 
            Mobile.Payment.Plan.Contract.Term,Contract_vs_non_contract))


ggplot(desc_file_model, aes(Contract_Length)) + stat_count()

#No contract Modelling data
model_data_NC <- desc_file_model %>%
  filter(Contract_Length == "0") %>%
  select(-Contract_Length)


#Contract Modelling data
moel_data_C <- desc_file_model %>%
  filter(Contract_Length != "0") %>%
  select(-Contract_Length)

#one-ho encode the 3 data sets. numerical varibales are ignored by default
desc_file_model_dum <- dummify(desc_file_model)
model_data_NC_dum <- dummify(model_data_NC)
model_data_C_dum <- dummify(moel_data_C)


#str(desc_file_model_dum)
#str(model_data_NC_dum)
#str(model_data_C_dum)



textab = paste("\n   Let's investigate which features are most correlated.\n",
               "      with different events of interest. This may hekp us in the \n",
               " step of feature selection cum analysis\n",
               "           let's find out")
ggplot() + 
  annotate("text", x = 4, y = 25, size=8, label = textab) + 
  theme_bw() +
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank())  + 
  theme(axis.title = element_blank()) + 
  theme(axis.text.x=element_blank()) + 
  theme(axis.text.y=element_blank())

#TARGETED (SUPERVISED) CORRELATIONS
desc_file_model_dum [,-8] %>%
  corr_cross()

model_data_C_dum [,-8] %>%
  corr_cross()


model_data_NC_dum [,-8] %>%
  corr_cross()


desc_file_model_dum [, -c(8, 52:54, 56:59)] %>%
  corr_var(event_group_Disconnect, top = 25)

model_data_C_dum[, -c(8, 49:51, 53:56)] %>%
  corr_var(event_group_Disconnect, top = 25)

model_data_NC_dum[, -c(8, 45:47, 49:52)] %>%
  corr_var(event_group_Disconnect, top = 25)


desc_file_model_dum[, -c(8,52, 53)] %>%
  corr_var(event_group_AS.IS, top = 40)

desc_file_model_dum[, -c(8,51,52)] %>%
  corr_var(event_group_Upgrade, top = 40)


#IMPUTING MISSING VALUES USING MLR

plot_missing(desc_file_model)
data_imputed <-  mlr::impute(obj = as.data.frame(desc_file_model), target = "event_group", classes = list(integer = imputeMedian(), numeric = imputeMean(), factor = imputeMode()), dummy.classes = "factor")

data_imputed_C <-  mlr::impute(obj = as.data.frame(moel_data_C), target = "event_group", classes = list(integer = imputeMedian(), numeric = imputeMean(), factor = imputeMode()), dummy.classes = "factor")

data_imputed_NC <-  mlr::impute(obj = as.data.frame(model_data_NC), target = "event_group", classes = list(integer = imputeMedian(), numeric = imputeMean(), factor = imputeMode()), dummy.classes = "factor")



data_imp <- data_imputed$data
data_imp_C <- data_imputed_C$data
data_imp_NC <- data_imputed_NC$data

plot_missing(data_imp)
plot_missing(data_imp_C)
plot_missing(data_imp_NC)

data_imp <-  data_imp %>%
  select (-cpl_month)
data_imp_C <-  data_imp_C %>%
  select (-cpl_month)
data_imp_NC <-  data_imp_NC %>%
  select (-cpl_month)


plot_missing(data_imp)
plot_missing(data_imp_C)
plot_missing(data_imp_NC)

data_immpp <- dummify(data_imp)
data_immpp_C <- dummify(data_imp_C)
data_immpp_NC <- dummify(data_imp_NC)
str(data_immpp)
data_split <- msplit(data_immpp, size = 0.7, seed = 0, print = T)
data_split_C <- msplit(data_immpp_C, size = 0.7, seed = 0, print = T)
data_split_NC <- msplit(data_immpp_NC, size = 0.7, seed = 0, print = T)

trainingset <- data_split$train
testingset <- data_split$test
trainingset_C <- data_split_C$train
testingset_C <- data_split_C$test
trainingset_NC <- data_split_NC$train
testingset_NC <- data_split_NC$test

#USING H2O-Traditional

h2o.init()


plt_md_data <- data_imp%>%
  ggplot(aes(event_group)) + stat_count(aes(fill = event_group))
plt_md_data


create_report(data_immpp)
str(trainingset)
str(trainingset_C)
str(trainingset_NC)
trainingset <- trainingset %>%
  drop_columns(c("AGE_BAND_NUM.dummy_TRUE","AGE_BAND_NUM.dummy_FALSE", "event_group_Upgrade", "event_group_AS.IS", "Plan_contract_Yes", "Plan_contract_No", "SUBS_ID"))
trainingset_C <- trainingset_C %>%
  drop_columns(c("SUBS_ID", "AGE_BAND_NUM.dummy_TRUE","AGE_BAND_NUM.dummy_FALSE", "event_group_Upgrade", "event_group_AS.IS", "Plan_contract_Yes", "Plan_contract_No" ))
trainingset_NC <- trainingset_NC %>%
  drop_columns(c("SUBS_ID", "AGE_BAND_NUM.dummy_TRUE","AGE_BAND_NUM.dummy_FALSE", "event_group_Upgrade", "event_group_AS.IS", "Plan_contract_Yes", "Plan_contract_No" , "upgrade_downgrage_upgraded", "upgrade_downgrage_unchanged", "upgrade_downgrage_Downgrade" ))
data_train <- as.h2o(trainingset)
data_test <- as.h2o(testingset)

str(data_train)
##DISCONNECT MODEL

seed = 123
set.seed(seed)
y <- "event_group_Disconnect"
x <- setdiff(names(data_train), y)
data_train[,y] <- as.factor(data_train[,y])
data_test[,y] <- as.factor(data_test[,y])

fitml_dscnct <- h2o.automl(x, y, training_frame = data_train,
                           nfolds = 5,
                           balance_classes = TRUE,
                           max_after_balance_size = 1,
                           max_runtime_secs = 3600, 
                           max_models = 20, 
                           stopping_metric = c( "AUTO"),
                           stopping_rounds = 3, seed = seed,
                           exclude_algos = "DeepLearning",
                           keep_cross_validation_predictions = TRUE,
                           sort_metric = c("AUTO"))

lb_dscnct <- fitml_dscnct@leaderboard


print(lb_dscnct, n = nrow(lb_dscnct))
fitml_dscnct@leader

perf_fitml_dscnt <- h2o.performance(fitml_dscnct@leader, newdata = data_test)

perf_fitml_dscnt

h2o.varimp_plot(h2o.getModel("GBM_3_AutoML_20190829_130740"), num_of_features = 20)

trainingset %>%
  select(upgrade_downgrage_Downgrade, DMAF, Contract_Length_0,
         SIMO_vs_nonSIMO_SIMO, Plan.Contract.Term, upgrade_downgrage_unchanged, upgrade_downgrage_upgraded, SIMO_vs_nonSIMO_NON_SIMO, Contract_Length_12, woy_cpl, doy_cpl,week_of_year, day_of_year,No_of_Services, face_val_22, face_val_40, AGE_BAND_NUM_25.to.34, Contract_Length_12, mpp_yes_no_No, AGE_BAND_NUM_18.to.24, event_group_Disconnect )%>%
  corr_var(event_group_Disconnect, top = 20)


#using modified h2o for faster run and plots

lare_fit_disconnect <- data_immpp[, -c(8, 40,42,46,47)] %>%
  h2o_automl("event_group_Disconnect",
             nfolds = 5,
             max_models = 4)


#model interpration
explainer <- dalex_explainer(
  df = filter(lare_fit_disconnect$datasets$global, train_test == "train"), 
  model = lare_fit_disconnect$model, 
  y = "event_group_Disconnect")
DMAF <- dalex_variable(explainer, variable = "DMAF")
plot(DMAF)
face_val_22 <- dalex_variable(explainer, variable = "face_val_22")
plot(face_val_22)

