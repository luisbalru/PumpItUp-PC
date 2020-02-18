import numpy as np
from sklearn.manifold import TSNE
import pandas as pd

import plotly.express as px
import plotly

import matplotlib.pyplot as plt

from preprocessing import preprocessing

from random import randint
colors = []

for i in range(3):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

X,y = preprocessing()
X = np.array(X)

reduced = TSNE(n_components=2).fit_transform(X)

cl0 = np.array([reduced[i] for i in range(len(reduced)) if y[i]=="functional"])
cl1 = np.array([reduced[i] for i in range(len(reduced)) if y[i]=="functional needs repair"])
cl2 = np.array([reduced[i] for i in range(len(reduced)) if y[i]=="non functional"])

print("Número de elementos de la clase 'functional' : " + str(len(cl0)))
print("Número de elementos de la clase 'functional needs repair' : " + str(len(cl1)))
print("Número de elementos de la clase 'non functional' : " + str(len(cl2)))

plt.scatter(cl0[:,0], cl0[:,1], color = colors[0], label = "Functional")
plt.scatter(cl1[:,0], cl1[:,1], color = colors[1], label = "Functional needs repair")
plt.scatter(cl2[:,0], cl2[:,1], color = colors[2], label = "Non functional")
plt.legend()
plt.show()

reduced = TSNE(n_components=3).fit_transform(X)

d = pd.DataFrame({"x": reduced[:,0], "y": reduced[:,1], "z": reduced[:,2], "labels": y})
fig = px.scatter_3d(d, x="x", y="y", z="z", color="labels")
fig.update_traces(marker=dict(size=5,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
plotly.offline.plot(fig, filename="plot_ohe_3d.html", auto_open=True)
