################################################
# RIPPER: PREPROCESAMIENTO Y CLASIFICACIÓN     #
# Autor: Luis Balderas Ruiz                    #
################################################

# Lectura de datos
train = read.csv("training.csv")
labels = read.csv("training-labels.csv")
labels = subset(labels, select = status_group )
test = read.csv("test.csv")
train = cbind(train,labels)



################################################
# EXPLORACIÓN DE DATOS


