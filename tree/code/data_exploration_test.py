import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

dataset = pd.read_csv("data/test.csv")

numeric_vars = dataset.select_dtypes(include=np.number)

categorical_vars = dataset.select_dtypes(exclude=np.number)

print("MISSING VALUES")
for col in dataset.columns:
    print("{}: {}".format(col, np.sum(pd.isna(dataset[col]))))

print()

print("CATEGORICAL COLUMNS - Nº OF CATEGORIES")
for col in categorical_vars:
    print("{}: {}".format(col, len(dataset[col].unique())))

corr = numeric_vars.corr()
cmap = sb.diverging_palette(220, 10, as_cmap=True)

mask = np.triu(np.ones_like(corr, dtype=np.bool))
sb.heatmap(abs(corr), mask=mask, cmap=cmap)

plt.show()
