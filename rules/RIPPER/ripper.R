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




################################################
# EXPLORACIÓN DE DATOS

# Balanceo de clases
table(train$status_group)
# Vemos que hay un gran desbalanceo
prop.table(table(train$status_group))

# Gráfica de barras por cantidad
qplot(quantity, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top")
qplot(status_group, data=train, geom="bar", fill=quantity) + 
  theme(legend.position = "top")


# Gráfica de barras para quality_group
qplot(quality_group, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top")

# Gráfica de barras para waterpoint_type
qplot(waterpoint_type, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top") + 
  theme(axis.text.x=element_text(angle = -20, hjust = 0))
#################################################################################
# Variables continuas

# Historgrama para construction_year agrupado por status_group
ggplot(train, aes(x = construction_year)) + 
  geom_histogram(bins = 20) + 
  facet_grid( ~ status_group)

# Agrupando con construction_year > 0
ggplot(subset(train, construction_year > 0), aes(x =construction_year)) +
  geom_histogram(bins = 20) + 
  facet_grid( ~ status_group)


################################################


################################################
# PREPROCESAMIENTO

# Limpieza de datos

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
table(data$permit)

# public_meeting
table(data$public_meeting)
data$public_meeting = as.character(data$public_meeting)
data$public_meeting[data$public_meeting == ''] = 'desconocido'
data$public_meeting = as.factor(data$public_meeting)
table(data$public_meeting)

# waterpoint_type_group
tabla_wtg = table(data$waterpoint_type_group)
data$waterpoint_type_group = as.character(data$waterpoint_type_group)
data$waterpoint_type_group[data$waterpoint_type_group == 'dam'] = 'other'
minoritarios = names(tabla_wtg[tabla_wtg<=150])
data$waterpoint_type_group[data$waterpoint_type_group %in% minoritarios] = 'other'
data$waterpoint_type_group = as.factor(data$waterpoint_type_group)
table(data$waterpoint_type_group)

# funder
tabla_funder = table(data$funder)
data$funder = as.character(data$funder)
data$funder[data$funder == '' | data$funder == 0] = 'desconocido'
minoritarios = names(tabla_funder[tabla_funder<=150])
data$funder[data$funder %in% minoritarios] = "otros"
data$funder = as.factor(data$funder)
table(data$funder)

# basin
table(data$basin)

#installer
tabla_installer = table(data$installer)
data$installer = as.character(data$installer)
data$installer[data$installer == '' | data$installer == 0 | data$installer == '-'] = 'desconocido'
minoritarios = names(tabla_installer[tabla_installer<=150])
data$installer[data$installer %in% minoritarios] = "otros"
data$installer = as.factor(data$installer)
table(data$installer)

# wpt_name --> no MV
table(data$wpt_name)

# basin --> no MV
table(data$basin)

# region
table(data$region)

# region code
table(data$region_code)

# lga
table(data$lga)

# ward
tabla_ward = table(data$ward)
data$ward = as.character(data$ward)
minoritarios = names(tabla_ward[tabla_ward<=60])
data$ward[data$ward %in% minoritarios] = "otros"
data$ward = as.factor(data$ward)
table(data$ward)

# scheme_name
tabla_sn = table(data$scheme_name)
data$scheme_name = as.character(data$scheme_name)
minoritarios = names(tabla_sn[tabla_sn<=200])
data$scheme_name[data$scheme_name %in% minoritarios] = "otros"
data$scheme_name = as.factor(data$scheme_name)
table(data$scheme_name)

################################################
# TRATAMIENTO DEL DESBALANCEO
# Balanceo de clases
test = data %>% filter(status_group == "")
test$status_group = NULL
train = data %>% filter(status_group != "")
train$id = NULL
train$status_group = as.character(train$status_group)
train$status_group = as.factor(train$status_group)
table(train$status_group)
# Vemos que hay un gran desbalanceo
prop.table(table(train$status_group))

install.packages("NoiseFiltersR")
library(NoiseFiltersR)
# IPF
salida_ipf = IPF(status_group~amount_tsh+latitude+longitude+date_recorded+basin+lga+funder+population+antiguedad+
                   gps_height+public_meeting+scheme_name+permit+extraction_type_class+management+
                   management_group+payment+quality_group+quantity+source+source_type+ source_class+
                   waterpoint_type, data = train)

#LVW
install.packages("FSinR")
library(FSinR)
resamplingParams <- list(method = "cv", number = 10) # Values for the caret trainControl function
fittingParams <- list(metric="Accuracy")
wrapper <- wrapperGenerator("JRip", resamplingParams, fittingParams) # wrapper method
salida_ipf$cleanData$status_group = as.character(salida_ipf$cleanData$status_group)
salida_ipf$cleanData$status_group = as.factor(salida_ipf$cleanData$status_group)
salida_ipf$cleanData$recorded_by = NULL
salida_lvw = lvw(salida_ipf$cleanData,'status_group',wrapper,K=5,verbose=TRUE)

# SMOTE
install.packages("DMwR")
library(DMwR)
table(salida_ipf$cleanData$status_group)
salida_smote = SMOTE(status_group~amount_tsh+latitude+longitude+date_recorded+basin+lga+funder+population+antiguedad+
                       gps_height+public_meeting+scheme_name+permit+extraction_type_class+management+
                       management_group+payment+quality_group+quantity+source+source_type+ source_class+
                       waterpoint_type, data = salida_ipf$cleanData, perc.over=10000,perc.under=5000)

################################################

################################################
# Correlación
# No se muestran correlaciones entre las variables numéricas
variables_numericas = subset(data, select=c(amount_tsh, gps_height, longitude, latitude))
cor(variables_numericas)





################################################
# CLASIFICACIÓN

inicio_test = nrow(train) + 1
fin = nrow(data)
train = data[1:(inicio_test-1),]
test = data[inicio_test:fin,]


# INTENTO 1. TODAS LAS VARIABLES. ERROR DE MEMORIA EN HEAP. Necesario seleccionar variables

model.Ripper1 = JRip(status_group~.-id, train)
summary(model.Ripper1)
model.Ripper1.pred = predict(model.Ripper1,newdata = test)

generaSubida("1",test$id,model.Ripper1.pred)


# INTENTO 2. QUITO VARIABLES CON MÁS OUTLIERS SIN POSIBILIDAD DE ARREGLAR Y LAS REPETIDAS (A MI JUICIO) --> 0.7420 

model.Ripper2 = JRip(status_group~amount_tsh+latitude+longitude+basin+lga+ward+region+population+antiguedad+
                      gps_height+public_meeting+scheme_management+permit+extraction_type_class+
                      management_group+quality_group+quantity_group+source_type+ source_class+
                      waterpoint_type_group, train)

summary(model.Ripper2)


model.Ripper2.pred = predict(model.Ripper2, newdata = test)
submission_int2 = data.frame(test$id)
submission_int2$status_group = model.Ripper2.pred
colnames(submission_int2) = c("id", "status_group")
write.csv(submission_int2,file="submission_int2.csv", row.names = FALSE)

# INTENTO 3. MODIFICACIÓN EN UNA VARIABLE --> 0.7455 

model.Ripper3 = JRip(status_group~amount_tsh+latitude+longitude+basin+region+population+antiguedad+
                       gps_height+public_meeting+scheme_management+permit+extraction_type_class+
                       management_group+quality_group+quantity_group+source_type+ source_class+
                       waterpoint_type_group, train)

summary(model.Ripper3)
model.Ripper3.pred = predict(model.Ripper3,newdata = test)

generaSubida("3",test$id,model.Ripper3.pred)

# INTENTO 4. REDUCCIÓN DRÁSTICA DE VARIABLES --> Peor resultado: 0.7325

model.Ripper4 = JRip(status_group~amount_tsh+latitude+longitude+population+antiguedad+
                       gps_height+extraction_type_class+quantity_group+waterpoint_type_group, train)

summary(model.Ripper4)
model.Ripper4.pred = predict(model.Ripper4,newdata = test)

generaSubida("4",test$id,model.Ripper4.pred)

# INTENTO 5. RETOMANDO 3 Y PREPROCESANDO FUNDER -->0.7457

model.Ripper5 = JRip(status_group~amount_tsh+latitude+longitude+basin+region+funder+population+antiguedad+
                       gps_height+public_meeting+scheme_management+permit+extraction_type_class+
                       management_group+quality_group+quantity_group+source_type+ source_class+
                       waterpoint_type_group, train)

summary(model.Ripper5)
model.Ripper5.pred = predict(model.Ripper5,newdata = test)

generaSubida("5",test$id,model.Ripper5.pred)

# INTENTO 6. RETOMANDO 5, ELIMINANDO PERMIT Y PUBLIC_MEETING --> 0.7436

model.Ripper6 = JRip(status_group~amount_tsh+latitude+longitude+basin+region+funder+population+antiguedad+
                       gps_height+scheme_management+permit+extraction_type_class+
                       management_group+quality_group+quantity_group+source_type+ source_class+
                       waterpoint_type_group, train)

summary(model.Ripper6)
model.Ripper6.pred = predict(model.Ripper6,newdata = test)

generaSubida("6",test$id,model.Ripper6.pred)

# INTENTO 7. RETOMANDO 3 Y PREPROCESANDO FUNDER --> 0.7521
      
model.Ripper7 = JRip(status_group~amount_tsh+latitude+longitude+date_recorded+basin+lga+funder+population+antiguedad+
                       gps_height+public_meeting+scheme_name+permit+extraction_type_class+management+
                       management_group+payment+quality_group+quantity+source+source_type+ source_class+
                       waterpoint_type, train)

summary(model.Ripper7)
model.Ripper7.pred = predict(model.Ripper7,newdata = test)

generaSubida("7",test$id,model.Ripper7.pred)


# INTENTO 8. A VER QUÉ PASA

model.Ripper8 = JRip(status_group~amount_tsh+latitude+longitude+ward+basin+lga+funder+population+antiguedad+
                       gps_height+public_meeting+scheme_management+permit+extraction_type_class+management+
                       management_group+payment+quality_group+quantity+source+source_type+ source_class+
                       waterpoint_type, train)
summary(model.Ripper8)
model.Ripper8.pred = predict(model.Ripper8,newdata = test)

generaSubida("8",test$id,model.Ripper8.pred)

# INTENTO 9

model.Ripper9 = JRip(status_group~amount_tsh+latitude+longitude+installer+basin+lga+funder+population+antiguedad+
                             gps_height+public_meeting+scheme_management+permit+extraction_type_class+management+
                             management_group+payment+quality_group+quantity+source+source_type+ source_class+
                             waterpoint_type, train)

summary(model.Ripper9)
model.Ripper9.pred = predict(model.Ripper9,newdata = test)

generaSubida("9",test$id,model.Ripper9.pred)

# INTENTO 10
model.Ripper10 = JRip(status_group~amount_tsh+latitude+longitude+date_recorded+installer+basin+lga+funder+population+antiguedad+
                       gps_height+public_meeting+scheme_name+permit+extraction_type_class+management+
                       management_group+payment+quality_group+quantity+source+source_type+ source_class+
                       waterpoint_type, train)

summary(model.Ripper10)
model.Ripper10.pred = predict(model.Ripper10,newdata = test)

generaSubida("10",test$id,model.Ripper10.pred)

# INTENTO 11 DE NUEVO CON 7

  