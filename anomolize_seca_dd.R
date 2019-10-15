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

seca_dd_cells <- sqlConnQuery("SELECT nename, localcellid, cellname FROM tmp.seca_dd_cells GROUP BY nename, localcellid, cellname", "tmp", -1)
seca_dd_dates <- sqlConnQuery("SELECT * FROM tmp.seca_dates ORDER BY starttime ASC", "tmp", -1)

# 20431 227696_Crossroads_3_UL 22
# 30995 303254_GlenWaverley_L1 21
# 40992 338725_Kilsyth2_2_UL 1
for(i in 40993:(length(seca_dd_cells$nename)-1)) {
  query <- paste("SELECT *, '' AS cellname FROM tmp.seca_dd WHERE nename='", seca_dd_cells$nename[i+1], "' AND localcellid='", seca_dd_cells$localcellid[i+1], "' GROUP BY starttime", sep="")
  seca_dd <- sqlConnQuery(query, "tmp", -1)
  
  cellname <- seca_dd_cells$cellname[i+1]
  
  if(is.null(cellname)) {
    cellname="Unknown"
  }
  
  seca_dd$cellname <- cellname
  
  seca_dd$starttime <- as.Date(seca_dd$starttime)
  seca_dd_dates$starttime <- as.Date(seca_dd_dates$starttime)
  
  # Check if all dates are present
  missing_dd <- setdiff(seca_dd_dates$starttime, seca_dd$starttime)
  
  if(length(missing_dd)>0) {
    
    for(j in 0:(length(missing_dd)-1)) {
      
      seca_dd_missing <- data.frame(starttime=as.Date(missing_dd[j+1], origin="1970-01-01"),
                                yyyy=format(as.Date(missing_dd[j+1], origin="1970-01-01"), format="%Y"),
                                mm=format(as.Date(missing_dd[j+1], origin="1970-01-01"), format="%m"),
                                dd=format(as.Date(missing_dd[j+1], origin="1970-01-01"), format="%d"),
                                weekday=format(as.Date(missing_dd[j+1], origin="1970-01-01"), format="%W"),
                                nename=seca_dd_cells$nename[i+1],
                                localcellid=seca_dd_cells$localcellid[i+1],
                                cellname=cellname,
                                rrc_connreq_att=-1,
                                dl_mac_mb=-1,
                                thrp_bits_ul=-1,
                                traffic_user_avg=-1,
                                traffic_user_max=-1,
                                hho_interenb_interfr_att=-1,
                                ra_ta_ue_index3=-1,
                                traffic_user_dl_qci_1=-1,
                                traffic_user_dl_qci_8=-1,
                                traffic_user_dl_qci_9=-1,
                                prb_dl_uti=-1)
      seca_dd <- rbind(seca_dd, seca_dd_missing)
    }
  }
  
  seca_dd$starttime <- as.Date(seca_dd$starttime)
  seca_dd <- seca_dd[order(seca_dd$starttime),]
  
  seca_dd_tbl_time <- as_tbl_time(seca_dd, starttime)
  seca_dd_anomalized <- seca_dd_tbl_time %>% time_decompose(traffic_user_avg, method = 'stl', merge = TRUE) %>% anomalize(remainder) %>% time_recompose()
  
  tmp <- dbConnect(MySQL(), user=usr, password=psswd, dbname='tmp', host=hst)
  on.exit(dbDisconnect(tmp))
  dbWriteTable(tmp, value=data.frame(seca_dd_anomalized), name="seca_dd_norm", append=TRUE, row.names=FALSE)
  dbDisconnect(tmp)
  
  print(paste(i, seca_dd_cells$nename[i+1], seca_dd_cells$localcellid[i+1], sep=" "))
}
