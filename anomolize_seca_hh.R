options(warn=-1)

library(RMySQL)
library(anomalize)
library(tibbletime)
library(dplyr)

usr <- .rs.askForPassword("Username:")
psswd <- .rs.askForPassword("Password:")
hst <- .rs.askForPassword("Host:")

# Returns dataframe given SQL String
sqlConnQuery <- function(sqlString, type, limit) {
  
  if(type=="tmp") {
    mydb = dbConnect(MySQL(), user=usr, password=psswd, dbname='tmp', host=hst)
  }
  
  result<-dbSendQuery(mydb, sqlString)
  data <- fetch(result, n=limit)
  huh <- dbHasCompleted(result)
  dbClearResult(result)
  dbDisconnect(mydb)
  return(data)
}

#fresult <- read.csv(file="/Users/61072380/Downloads/seca_hh_anomaly_cells.csv", header=TRUE, sep=",")
seca_hh_cells <- read.csv(file="/Users/61072380/Desktop/seca_hh_dupes.csv", header=TRUE, sep=",")
#seca_hh_cells <- data.frame(fresult[[2]])
#colnames(seca_hh_cells) <- "cellname"
#seca_hh_cells <- sqlConnQuery("SELECT cellname FROM tmp.seca_hh_cells WHERE count<336 AND count>47 GROUP BY cellname", "tmp", -1)
seca_hh_hours <- sqlConnQuery("SELECT starttime FROM tmp.seca_hours ORDER BY starttime ASC", "tmp", -1)

# 403 240323_Hallsville_MS_L08B_1
for(i in 404:(length(seca_hh_cells$cellname)-1)) {
  query <- paste("SELECT * FROM tmp.seca_hh WHERE cellname='", seca_hh_cells$cellname[i+1], "' AND traffic_user_avg>=0 GROUP BY starttime", sep="")
  seca_hh <- sqlConnQuery(query, "tmp", -1)
  
  seca_hh$starttime <- as.POSIXlt(seca_hh$starttime, format='%Y-%m-%d %H:%M:%S')
  seca_hh_hours$starttime <- as.POSIXlt(seca_hh_hours$starttime, format='%Y-%m-%d %H:%M:%S')
  
  # Check if all timestamps are present
  missing_hh <- setdiff(seca_hh_hours$starttime, seca_hh$starttime)
  
  if(length(missing_hh)>0) {
    
    for(j in 0:(length(missing_hh)-1)) {
      
      seca_hh_missing <- data.frame(starttime=as.POSIXlt(missing_hh[j+1], origin="1970-01-01", format='%Y-%m-%d %H:%M:%S'),
                                    yyyy=format(as.Date(missing_hh[j+1], origin="1970-01-01"), format="%Y"),
                                    mm=format(as.Date(missing_hh[j+1], origin="1970-01-01"), format="%m"),
                                    dd=format(as.Date(missing_hh[j+1], origin="1970-01-01"), format="%d"),
                                    hh=as.POSIXlt(missing_hh[j+1], origin="1970-01-01", format='%H'),
                                    weekday=format(as.Date(missing_hh[j+1], origin="1970-01-01"), format="%W"),
                                    cellname=seca_hh_cells$cellname[i+1],
                                    rrc_connreq_att=-1,
                                    dl_mac_mb=-1,
                                    thrp_bits_ul=-1,
                                    traffic_user_avg=-1,
                                    traffic_user_max=-1,
                                    hho_interenb_interfr_att=-1,
                                    ra_ta_ue_index3=-1,
                                    traffic_activeuser_dl_qci_1=-1,
                                    traffic_activeuser_dl_qci_8=-1,
                                    traffic_activeuser_dl_qci_9=-1,
                                    traffic_activeuser_dl_avg=-1,
                                    prb_dl_uti=-1)
      seca_hh <- rbind(seca_hh, seca_hh_missing)
    }
  }
  
  seca_hh$starttime <- as.POSIXct(seca_hh$starttime, format='%Y-%m-%d %H:%M:%S')
  seca_hh <- seca_hh[order(seca_hh$starttime),]
  
  seca_hh_tbl_time <- as_tbl_time(seca_hh, starttime)
  seca_hh_anomalized <- seca_hh_tbl_time %>% time_decompose(traffic_user_avg, method = 'stl', merge = TRUE) %>% anomalize(remainder) %>% time_recompose()
  
  tmp <- dbConnect(MySQL(), user=usr, password=psswd, dbname='tmp', host=hst)
  on.exit(dbDisconnect(tmp))
  dbWriteTable(tmp, value=data.frame(seca_hh_anomalized), name="seca_hh_norm", append=TRUE, row.names=FALSE)
  dbDisconnect(tmp)
  
  print(paste(i, seca_hh_cells$cellname[i+1], sep=" "))
}
