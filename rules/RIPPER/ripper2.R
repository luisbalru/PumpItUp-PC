################################################
# RIPPER: PREPROCESAMIENTO Y CLASIFICACIÓN     #
# Autor: Luis Balderas Ruiz                    #
################################################

library(RWeka)
library(ggplot2)
library(rpart)
library(dplyr)


###############################################
# FUNCIONES PROPIAS
generaSubida = function(numero, test_id, prediccion){
  nombre = paste("submission_int",numero,".csv", sep="")
  submission = data.frame(test_id)
  submission$status_group = prediccion
  colnames(submission) = c("id", "status_group")
  write.csv(submission,file=nombre, row.names = FALSE)
  
}

# Lectura de datos
train = read.csv("training.csv")
labels = read.csv("training-labels.csv")
test = read.csv("test.csv")
train = merge(train, labels)

##############################################
# PREPROCESAMIENITO

test$status_group = ""
data = rbind(train,test)

data$num_private = NULL
data$recorded_by = NULL
data$wpt_name = NULL
data$extraction_type = NULL
data$extraction_type_group = NULL
data$payment_type = NULL
data$water_quality = NULL
data$scheme_management = NULL
data$district_code = NULL
data$region<-NULL
data$region_code<-NULL
data$subvillage<-NULL
data$ward<- NULL
data$waterpoint_type_group<-NULL
data$quantity_group<-NULL
data$installer<-NULL
data$date_recorded = as.Date(data$date_recorded)
data$amount_tsh[data$amount_tsh>=10000] = 10000

mv_gps = rpart(gps_height ~ latitude + longitude, data = data[(data$gps_height>0),], method = "anova")
data$gps_height[data$gps_height <= 0] = predict(mv_gps,data[(data$gps_height<=0),])

data$longitude[data$region == "Arusha" & data$longitude == 0] =	36.55407
data$longitude[data$region=="Dar es Salaam" & data$longitude==0] = 39.21294
data$longitude[data$region=="Dodoma" & data$longitude==0] = 36.04196
data$longitude[data$region=="Iringa" & data$longitude==0] = 34.89592
data$longitude[data$region=="Kagera" & data$longitude==0] = 31.23309
data$longitude[data$region=="Kigoma" & data$longitude==0] = 30.21889
data$longitude[data$region=="Kilimanjaro" & data$longitude==0] = 37.50546
data$longitude[data$region=="Lindi" & data$longitude==0] = 38.98799
data$longitude[data$region=="Manyara" & data$longitude==0] = 35.92932
data$longitude[data$region=="Mara" & data$longitude==0] = 34.15698
data$longitude[data$region=="Mbeya" & data$longitude==0] = 33.53351
data$longitude[data$region=="Morogoro" & data$longitude==0] = 37.04678
data$longitude[data$region=="Mtwara" & data$longitude==0] = 39.38862
data$longitude[data$region=="Mwanza" & data$longitude==0] = 33.09477
data$longitude[data$region=="Pwani" & data$longitude==0] = 38.88372
data$longitude[data$region=="Rukwa" & data$longitude==0] = 31.29116
data$longitude[data$region=="Ruvuma" & data$longitude==0] = 35.72784
data$longitude[data$region=="Shinyanga" & data$longitude==0] = 33.24037
data$longitude[data$region=="Singida" & data$longitude==0] = 373950
data$longitude[data$region=="Tabora" & data$longitude==0] = 32.87830
data$longitude[data$region=="Tanga" & data$longitude==0] = 38.50195

data$population[data$population>5000] = 5000
data$antiguedad = max(data$construction_year) - data$construction_year
data$antiguedad[data$antiguedad == max(data$construction_year)] = round(median(data$antiguedad[data$antiguedad != max(data$construction_year)]),digits = 0)

# permit --> introduzco desconocido
table(data$permit)
data$permit = as.character(data$permit)
data$permit[data$permit == ''] = 'desconocido'
data$permit = as.factor(data$permit)
table(data$permit)

# public_meeting
table(data$public_meeting)
data$public_meeting = as.character(data$public_meeting)
data$public_meeting[data$public_meeting == ''] = 'desconocido'
data$public_meeting = as.factor(data$public_meeting)
table(data$public_meeting)

# funder
tabla_funder = table(data$funder)
data$funder = as.character(data$funder)
data$funder[data$funder == '' | data$funder == 0] = 'desconocido'
minoritarios = names(tabla_funder[tabla_funder<=150])
data$funder[data$funder %in% minoritarios] = "otros"
data$funder = as.factor(data$funder)
table(data$funder)

# scheme_name
tabla_sn = table(data$scheme_name)
data$scheme_name = as.character(data$scheme_name)
data$scheme_name[data$scheme_name == ''] = "otros"
minoritarios = names(tabla_sn[tabla_sn<=200])
data$scheme_name[data$scheme_name %in% minoritarios] = "otros"
data$scheme_name = as.factor(data$scheme_name)
table(data$scheme_name)

install.packages("ROSE")
library(ROSE)
train_pr = data %>% filter(status_group != '')
test_pr = data %>% filter(status_group == '')
test_pr$status_group = NULL
train_pr$class = train_pr$status_group
train_pr$class = as.character(train_pr$class)
train_pr$class[train_pr$status_group == "functional" | train_pr$status_group == "non functional"] = 'mayoritaria'
train_pr$class = as.factor(train_pr$class)
train_pr$id=NULL
test_pr$id=NULL

oversampling = ovun.sample(class~.-status_group, p=0.4, train_pr,seed=77145416)
resultado_ovun = oversampling$data
resultado_ovun$class = NULL
undersampling = ovun.sample(class~.-status_group, p=0.4, method="under", train_pr,seed=77145416)
resultado_ovun2 = undersampling$data

resultado_ovun2$class = NULL
resultado_ovun2$status_group = as.character(resultado_ovun2$status_group)
resultado_ovun2$status_group = as.factor(resultado_ovun2$status_group)

resultado_ovun$status_group = as.character(resultado_ovun$status_group)
resultado_ovun$status_group = as.factor(resultado_ovun$status_group)

# DUMMIES PARA ENN
install.packages("fastDummies")
library(fastDummies)
rovun_dummies = dummy_cols(resultado_ovun)
rovun_dummies$date_recorded = NULL
rovun_dummies$funder = NULL
rovun_dummies$lga = NULL
rovun_dummies$basin = NULL
rovun_dummies$public_meeting = NULL
rovun_dummies$scheme_name = NULL
rovun_dummies$permit = NULL
rovun_dummies$extraction_type_class = NULL
rovun_dummies$management = NULL
rovun_dummies$management_group = NULL
rovun_dummies$payment = NULL
rovun_dummies$quality_group = NULL
rovun_dummies$quantity = NULL
rovun_dummies$source = NULL
rovun_dummies$source_class = NULL
rovun_dummies$waterpoint_type = NULL
rovun_dummies$source_type = NULL
library(NoiseFiltersR)

range01 = function(x){
  return((x-min(x))/(max(x)-min(x)))
}
rovun_dummies[,1:5] = apply(rovun_dummies[,1:5],2,range01)
# res_enn = ENN(status_group~.,data=rovun_dummies,k=5)
# Paso a Python
write.csv(rovun_dummies,"rovun_dummies.csv")


###############################################################################
# INTENTO 13
modelo.Ripper13 = JRip(status_group~.,data = resultado_ovun)
summary(modelo.Ripper13)
modelo.Ripper13.pred = predict(modelo.Ripper13,newdata = test_pr)

generaSubida("13",test$id,modelo.Ripper13.pred)

# INTENTO 14
resultado_ovun$amount_tsh = NULL

modelo.Ripper14 = JRip(status_group~.,data = resultado_ovun)
summary(modelo.Ripper14)
modelo.Ripper14.pred = predict(modelo.Ripper14,newdata = test_pr)

generaSubida("14",test$id,modelo.Ripper14.pred)

# INTENTO 15: UNDERSAMPLING no va
modelo.Ripper15 = JRip(status_group~., data=resultado_ovun2)
summary(modelo.Ripper15)

# INTENTO 16 
resultado_ovun$date_recorded = NULL

modelo.Ripper16 = JRip(status_group~.,data = resultado_ovun)
summary(modelo.Ripper16)
modelo.Ripper16.pred = predict(modelo.Ripper16,newdata = test_pr)

generaSubida("16",test$id,modelo.Ripper16.pred)
