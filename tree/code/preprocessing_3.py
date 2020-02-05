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

train_dataset = pd.merge(train_dataset, train_labels)
fill_values = {}

for col in train_dataset.columns:
    if col != "status_group":
        if np.issubdtype(train_dataset[col].dtype, np.number):
            fill_val_1 = np.mean(train_dataset.loc[
                train_dataset['status_group'] == "functional", col
            ])
            fill_val_2 = np.mean(train_dataset.loc[
                train_dataset['status_group'] == "non functional", col
            ])
            fill_val_3 = np.mean(train_dataset.loc[
                train_dataset['status_group'] == "functional needs repair", col
            ])
            train_dataset.loc[
                train_dataset['status_group'] == "functional", col
            ] = train_dataset.loc[
                train_dataset['status_group'] == "functional", col
            ].fillna(fill_val_1)
            train_dataset.loc[
                train_dataset['status_group'] == "non functional", col
            ] = train_dataset.loc[
                train_dataset['status_group'] == "non functional", col
            ].fillna(fill_val_2)
            train_dataset.loc[
                train_dataset['status_group'] == "functional needs repair", col
            ] = train_dataset.loc[
                train_dataset['status_group'] == "functional needs repair", col
            ].fillna(fill_val_3)

            fill_val = np.mean(train_dataset[col])
        else:
            fill_val_1 = train_dataset.loc[
                train_dataset['status_group'] == "functional", col
            ].mode()[0]
            fill_val_2 = train_dataset.loc[
                train_dataset['status_group'] == "non functional", col
            ].mode()[0]
            fill_val_3 = train_dataset.loc[
                train_dataset['status_group'] == "functional needs repair", col
            ].mode()[0]
            train_dataset.loc[
                train_dataset['status_group'] == "functional", col
            ] = train_dataset.loc[
                train_dataset['status_group'] == "functional", col
            ].fillna(fill_val_1)
            train_dataset.loc[
                train_dataset['status_group'] == "non functional", col
            ] = train_dataset.loc[
                train_dataset['status_group'] == "non functional", col
            ].fillna(fill_val_2)
            train_dataset.loc[
                train_dataset['status_group'] == "functional needs repair", col
            ] = train_dataset.loc[
                train_dataset['status_group'] == "functional needs repair", col
            ].fillna(fill_val_3)

            fill_val = train_dataset[col].mode()[0]

        fill_values[col] = fill_val

test_dataset = test_dataset.fillna(value=fill_values)

train_dataset.to_csv("train-preprocessed.csv")
test_dataset.to_csv("test-preprocessed.csv")
