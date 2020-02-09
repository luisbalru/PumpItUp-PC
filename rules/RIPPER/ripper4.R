################################################
# RIPPER: PREPROCESAMIENTO Y CLASIFICACIÓN     #
# Autor: Luis Balderas Ruiz                    #
################################################
options(java.parameters = "-Xmx55g")
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



# EXPLORACIÓN ...

################################################
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

# Estudio de amount_sh
summary(data$amount_tsh)
summary(data$amount_tsh[data$amount_tsh>0])
hist(data$amount_tsh)
# Apenas hay datos con amount_sh > 10000, luego esos valores pueden incluso despistar
table(data$status_group[data$amount_tsh>10000],data$amount_tsh[data$amount_tsh>10000])
# Por tanto, los valores mayores que 10000 parecen ser outliers, por lo que
# les doy el valor de 10000
data$amount_tsh[data$amount_tsh>=10000] = 10000
hist(data$amount_tsh)
hist(data$amount_tsh[data$amount_tsh>=5000])
# Parece haber una gran frecuencia entorno al 0, dato que no tendría sentido
length(data$amount_tsh[data$amount_tsh==0]) # 52049 casos de 0
length(train$amount_tsh[train$amount_tsh == 0]) #41639 en train
length(test$amount_tsh[test$amount_tsh == 0]) # 10410 en test

# Trabajo con fechas. Transformamos a fechas formales
data$date_recorded = as.Date(data$date_recorded)

# Estudio de gps_height
hist(data$gps_height)
summary(data$gps_height)
length(data$gps_height[data$gps_height == 0]) # 25649
length(data$gps_height[data$gps_height < 0]) # 1881 casos menores de 0

# Defino un árbol de decisión para rellenar los valores perdidos (<=0) de gps_height
# en función de la longitud y latitud del terreno
mv_gps = rpart(gps_height ~ latitude + longitude, data = data[(data$gps_height>0),], method = "anova")
data$gps_height[data$gps_height <= 0] = predict(mv_gps,data[(data$gps_height<=0),])
# Dada la orografía de Tanzania, tras la predicción tiene más sentido el resultado de la altura
hist(data$gps_height)
summary(data$gps_height)


# Estudio de latitude. Según Google Maps, debe estar entre -15 y 0.
summary(data$latitude)
hist(data$latitude)
length(data$latitude[data$latitude>=0])

# Estudio de longitude. Según Google Maps, debe estar en 30 y algo más de 40
summary(data$longitude)
hist(data$longitude)
length(data$longitude[data$longitude<30])

longitudes_medias = aggregate(longitude~region,data=data[(data$longitude!=0),], FUN=mean)

# Cambio los 0, que son claramente outliers o MV por las longitudes
# en media

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

hist(data$longitude)


# Estudio population
hist(data$population)
# Número de fuentes con nadie alrededor --> 26834
# Según las estadísticas, la población en Tanzania es de 67 personas/km². Parecen valores perdidos
length(data$population[data$population==0])
# Con más de 5000 personas
hist(data$population[data$population>5000])
# Apenas hay zonas con más de 10000 personas alrededor de la fuente
hist(data$population[data$population>10000])
# Utilizo la media para los MV
data$population[data$population==0] = round(mean(data$population[data$population!=0]),digits = 0)
hist(data$population)
table(data$population)
table(data$status_group[data$population>5000],data$population[data$population>5000])
# Concentro los valores superiores a 5000 en 5000
data$population[data$population>5000] = 5000
hist(data$population)

# Creación de la variable antiguedad para trabajar con construction_year
data$antiguedad = max(data$construction_year) - data$construction_year
table(data$antiguedad)
# La mayoría de las fuentes (25969) tienen 2013 años de antigüedad, es decir, construction_year contenía un 0 (MV). Por tanto,
# trabajo con la media de las antigüedades para solucionarlo
data$antiguedad[data$antiguedad == max(data$construction_year)] = round(median(data$antiguedad[data$antiguedad != max(data$construction_year)]),digits = 0)
hist(data$antiguedad)

# VARIABLES CATEGÓRICAS CON VALORES DISPARES

# scheme-management. Los minoritarios a Other
table(data$scheme_management)
data$scheme_management = as.character(data$scheme_management)
data$scheme_management[data$scheme_management == 'None'] = 'Other'
data$scheme_management[data$scheme_management == 'SWC'] = 'Other'
data$scheme_management[data$scheme_management == 'Trust'] = 'Other'
data$scheme_management[data$scheme_management == ''] = 'Other'
data$scheme_management = as.factor(data$scheme_management)
table(data$scheme_management)

# permit --> introduzco desconocido
table(data$permit)
data$permit = as.character(data$permit)
data$permit[data$permit == ''] = 'desconocido'
data$permit = as.factor(data$permit)

# public_meeting
table(data$public_meeting)
data$public_meeting = as.character(data$public_meeting)
data$public_meeting[data$public_meeting == ''] = 'desconocido'
data$public_meeting = as.factor(data$public_meeting)

# waterpoint_type_group
table(data$waterpoint_type_group)
data$waterpoint_type_group = as.character(data$waterpoint_type_group)
data$waterpoint_type_group[data$waterpoint_type_group == 'dam'] = 'other'
data$waterpoint_type_group = as.factor(data$waterpoint_type_group)

# funder
tabla_funder = table(data$funder)
data$funder = as.character(data$funder)
data$funder[data$funder == '' | data$funder == 0] = 'desconocido'
minoritarios = rownames(tabla_funder[tabla_funder<=150])
data$funder[data$funder %in% minoritarios] = "otros"
data$funder = as.factor(data$funder)
table(data$funder)

# basin
table(data$basin)

#installer
table(data$installer)
data$installer = as.character(data$installer)
data$installer[data$installer == '' | data$installer == 0 | data$installer == '-'] = 'desconocido'
data$installer = as.factor(data$installer)
table(data$installer)
