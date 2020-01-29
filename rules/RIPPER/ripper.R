################################################
# RIPPER: PREPROCESAMIENTO Y CLASIFICACIÓN     #
# Autor: Luis Balderas Ruiz                    #
################################################

library(RWeka)
library(ggplot2)
library(dplyr)

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


################################################




################################################
# CLASIFICACIÓN


model.Ripper = JRip(status_group~., train)

summary(model.Ripper)


model.Ripper.pred = predict(model.Ripper, newdata = iris.test)
