import numpy as np
import pandas as pd

train_dataset = pd.read_csv("data/training.csv")
train_labels = pd.read_csv("data/training-labels.csv")
test_dataset = pd.read_csv("data/test.csv")

categorical_vars = train_dataset.select_dtypes(exclude=np.number)

## FEATURE ELIMINATION
variables_to_drop = [
    "scheme_name",
    "recorded_by",
    "region_code",
    "construction_year"
]

for col in categorical_vars.columns:
    if len(full_dataset[col].unique()) > 100:
        variables_to_drop.append(col)

train_dataset.drop(columns=variables_to_drop, inplace=True)
test_dataset.drop(columns=variables_to_drop, inplace=True)

fill_values = {}
for col in train_dataset.columns:
    if np.issubdtype(train_dataset[col].dtype, np.number):
        fill_val = np.mean(train_dataset[col])
    else:
        fill_val = train_dataset[col].mode()[0]

    fill_values[col] = fill_val

train_dataset = train_dataset.fillna(value=fill_values)
test_dataset = test_dataset.fillna(value=fill_values)

train_dataset = pd.merge(train_dataset, train_labels)

train_dataset.to_csv("train-preprocessed.csv")
test_dataset.to_csv("test-preprocessed.csv")
