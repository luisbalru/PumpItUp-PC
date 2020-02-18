################################################
# RIPPER: PREPROCESAMIENTO Y CLASIFICACIÓN     #
# Autor: Luis Balderas Ruiz                    #
################################################
#options(java.parameters = "-Xmx55g")
library(RWeka)
library(ggplot2)
library(rpart)
library(dplyr)


###############################################
# FUNCIONES PROPIAS

Accuracy = function(pred,etiq){
  return(length(pred[pred == etiq])/length(pred))
}

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

# PREPROCESAMIENTO

# Limpieza de datos

# Train --> construction_year

train$construction_year[train$construction_year == 0 & train$status_group == 'functional'] = round(mean(train$construction_year[train$construction_year != 0 & train$status_group == 'functional']))
train$construction_year[train$construction_year == 0 & train$status_group == 'non functional'] = round(mean(train$construction_year[train$construction_year != 0 & train$status_group == 'non functional']))
train$construction_year[train$construction_year == 0 & train$status_group == 'functional needs repair'] = round(mean(train$construction_year[train$construction_year != 0 & train$status_group == 'functional needs repair']))

antigua_subida = read.csv("new.csv")
test = cbind(test, status_group=antigua_subida$status_group)

test$construction_year[test$construction_year == 0 & test$status_group == 'functional'] = round(mean(test$construction_year[test$construction_year != 0 & test$status_group == 'functional']))
test$construction_year[test$construction_year == 0 & test$status_group == 'non functional'] = round(mean(test$construction_year[test$construction_year != 0 & test$status_group == 'non functional']))
test$construction_year[test$construction_year == 0 & test$status_group == 'functional needs repair'] = round(mean(test$construction_year[test$construction_year != 0 & test$status_group == 'functional needs repair']))


# Creación de la variable estado en el test para que
# coincidan en número a la hora de hacer transformaciones
test$status_group = ""

data = rbind(train,test)




data$wpt_name = NULL
data$subvillage = NULL
data$ward = NULL 
data$recorded_by = NULL
data$scheme_name = NULL
data$num_private = NULL
data$region_code = NULL
data$quantity_group = NULL
data$source_type = NULL
data$waterpoint_type_group = NULL
data$payment_type = NULL

data$funder = as.character(data$funder)
data$funder[data$funder == ''] = 'Other'
data$funder = as.factor(data$funder)

data$installer = as.character(data$installer)
data$installer[data$installer == ''] = 0
data$installer = as.factor(data$installer)

data$permit = as.character(data$permit)
data$permit[data$permit == ''] = 'True'
data$permit = as.factor(data$permit)

data$scheme_management = as.character(data$scheme_management)
data$scheme_management[data$scheme_management == ''] = 0
data$scheme_management = as.factor(data$scheme_management)

data$public_meeting = as.character(data$public_meeting)
data$public_meeting[data$public_meeting == ''] = 'True'
data$public_meeting = as.factor(data$public_meeting)

data$installer = as.character(data$installer)
data$installer[data$installer == 0 | data$installer == '-'] = 'Other'
data$installer = as.factor(data$installer)

data$gps_height[data$gps_height == 0] = 1166
data$date_recorded = as.Date(data$date_recorded)
data$extraction_type_group = NULL

data$extraction_type = as.character(data$extraction_type)
data$extraction_type[data$extraction_type == 'india mark ii'] = 'india'
data$extraction_type[data$extraction_type == 'india mark iii'] = 'india'
data$extraction_type[data$extraction_type == 'other - swn 81' | data$extraction_type == 'swn 80'] = 'swn'
data$extraction_type[data$extraction_type == 'walimi' | data$extraction_type == 'other - mkulima/shinyanga' | data$extraction_type == 'other - play pump'] = 'other handpump'
data$extraction_type[data$extraction_type == 'cemo' | data$extraction_type == 'climax'] = 'other motorpump'
data$extraction_type = as.factor(data$extraction_type)


inicio_test = nrow(train) + 1
fin = nrow(data)
train = data[1:(inicio_test-1),]
test = data[inicio_test:fin,]

model.Ripper25 = JRip(status_group~latitude+longitude+date_recorded+basin+lga+funder+population+construction_year+installer+
                        gps_height+public_meeting+scheme_management+permit+extraction_type+management+
                        management_group+payment+quality_group+quantity+source+ source_class+
                        waterpoint_type, train, control = Weka_control(F = 2, N=3,O=29))

summary(model.Ripper25)
model.Ripper25.pred = predict(model.Ripper25,newdata = test)

generaSubida('25',test$id,model.Ripper25.pred)

