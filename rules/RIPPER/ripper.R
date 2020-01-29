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

# Estudio de latitude. Según Google Maps, debe estar entre -15 y 0.
summary(data$latitude)
hist(data$latitude)
length(data$latitude[data$latitude>=0])

# Estudio de longitude. Según Google Maps, debe estar en 30 y algo más de 40
summary(data$longitude)
hist(data$longitude)
length(data$longitude[data$longitude<30])

resumen_longitudes <- aggregate(longitude~region,data=data[(data$longitude!=0),], FUN=mean)

################################################




################################################
# CLASIFICACIÓN


model.Ripper = JRip(status_group~., train)

summary(model.Ripper)


model.Ripper.pred = predict(model.Ripper, newdata = iris.test)
