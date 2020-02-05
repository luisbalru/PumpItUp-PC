import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

dataset = pd.read_csv("data/training.csv")
labels = pd.read_csv("data/training-labels.csv")
full_dataset = pd.merge(dataset, labels)

numeric_vars = full_dataset.select_dtypes(include=np.number)

numeric_vars = pd.merge(numeric_vars, labels)
numeric_vars.drop(columns="id", inplace=True)

categorical_vars = full_dataset.select_dtypes(exclude=np.number)

print("MISSING VALUES")
for col in full_dataset.columns:
    print("{}: {}".format(col, np.sum(pd.isna(full_dataset[col]))))

print()

print("CATEGORICAL COLUMNS - Nº OF CATEGORIES")
for col in categorical_vars:
    print("{}: {}".format(col, len(full_dataset[col].unique())))

corr = numeric_vars.corr()
cmap = sb.diverging_palette(220, 10, as_cmap=True)

mask = np.triu(np.ones_like(corr, dtype=np.bool))
sb.heatmap(abs(corr), mask=mask, cmap=cmap)


plt.show()
