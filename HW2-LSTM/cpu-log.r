library(dplyr)
library(chron)
library(ggplot2)
data=read.csv("Memory.csv")
tail(data)
names(data)<-c("machine_id","date","timestamp","Memory_ghz","ip","Memory_usage","total")
data[,3] <- sapply(data[,3], as.character)
data[,3] <- as.POSIXct(data[,3], format="%Y-%m-%d %H:%M:%S")
outdata<-(data %>%group_by(machine_id,cut(data[,3], breaks="30 sec")) %>%summarize(Memory_ghz_max = max(Memory_ghz),Memory_usage_max = max(Memory_usage)))
maxs <- aggregate(list(data$Memory_usage,data$Memory_ghz), 
                  list(data$machine_id,cut(data[,3], breaks="30 sec")),
                  max)
outdata

names(maxs)<-c("machine","time","Memory_usage_max","Memory_ghz_max")
head(maxs)
i325376172<-outdata[outdata[1]=="i-325376172-UserCluster1-sysadmin",]
a957043145<-outdata[outdata[1]=="a-957043145-UserCluster1-sysadmin",]
b956223090<-outdata[outdata[1]=="b-956223090-UserCluster1-sysadmin",]
c959255288<-outdata[outdata[1]=="c-959255288-UserCluster1-sysadmin",]
t657740490<-outdata[outdata[1]=="t-657740490-UserCluster1-sysadmin",]
z323389049<-outdata[outdata[1]=="z-323389049-UserCluster1-sysadmin",]
#qplot(seq(1,to=length(c959255288[[4]])),c959255288[[4]],geom="line")
df=i325376172
getstate<-function(df){
  df$mean_Memory_usage_max<-mean(df$Memory_usage_max)
  df$sd_Memory_usage_max<-sd(df$Memory_usage_max)
  df$state<-0
  df[df$Memory_usage_max>df$mean_Memory_usage_max+df$sd_Memory_usage_max*1,'state']<-1
  df[df$Memory_usage_max>df$mean_Memory_usage_max+df$sd_Memory_usage_max*2,'state']<-2
  df[df$Memory_usage_max<df$mean_Memory_usage_max-df$sd_Memory_usage_max*1,'state']<--1
  df[df$Memory_usage_max<df$mean_Memory_usage_max-df$sd_Memory_usage_max*2,'state']<--2
  table(df$state)
  return(df[1:11000,])
  
}

getvibrate<-function(df){
  df$vibrate<-0
  for (i in 1:10999){
    df[i,'vibrate']<-df[i+1,'Memory_usage_max']/df[i,'Memory_usage_max']-1
  }
  df$vibrateState<-0
  df$vibrateup_mean<-mean(df[df$vibrate>0,]$vibrate)
  df$vibrateup_sd<-sd(df[df$vibrate>0,]$vibrate)
  df$vibratedown_mean<-mean(df[df$vibrate<0,]$vibrate)
  df$vibratedown_sd<-sd(df[df$vibrate<0,]$vibrate)
  df[df$vibrate>(df$vibrateup_mean+df$vibrateup_sd/6),]$vibrateState<-1
  #df[df$vibrate>(df$vibrateup_mean+2*df$vibrateup_sd),]$vibrateState<-2
  #df[df$vibrate>(df$vibrateup_mean+3*df$vibrateup_sd),]$vibrateState<-3
  #df[df$vibrate>(df$vibrateup_mean+4*df$vibrateup_sd),]$vibrateState<-4
  df[df$vibrate<(df$vibratedown_mean-1*df$vibratedown_sd/6),]$vibrateState<--1
  #df[df$vibrate<(df$vibratedown_mean-2*df$vibratedown_sd),]$vibrateState<--2
  #df[df$vibrate<(df$vibratedown_mean-3*df$vibratedown_sd),]$vibrateState<--3
  #df[df$vibrate<(df$vibratedown_mean-4*df$vibratedown_sd),]$vibrateState<--4
  table(df$vibrateState)
  return(df)
}
machines<-list(i325376172,a957043145,b956223090,c959255288,t657740490,z323389049)
machines<-lapply(machines, getstate)
machines<-lapply(machines, getvibrate)



bind_machines<-rbind(machines[[1]],machines[[2]],machines[[3]],machines[[4]],machines[[5]],machines[[6]])
View(bind_machines)


outdata<-bind_machines[,c(3,4,7,8,9)]
write.csv(outdata,"outdata.csv")





