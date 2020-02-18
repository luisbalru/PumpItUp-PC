import numpy as np
import pandas as pd

train_dataset = pd.read_csv("../../data/training.csv")
train_labels = pd.read_csv("../../data/training-labels.csv")
test_dataset = pd.read_csv("../../data/test.csv")

train_dataset['date_recorded'] = pd.to_datetime(train_dataset['date_recorded'])
test_dataset['date_recorded'] = pd.to_datetime(test_dataset['date_recorded'])

train_dataset['year_recorded'] = train_dataset['date_recorded'].map(
    lambda x: x.year
)

test_dataset['year_recorded'] = test_dataset['date_recorded'].map(
    lambda x: x.year
)

train_dataset['month_recorded'] = train_dataset['date_recorded'].map(
    lambda x: x.month
)

test_dataset['month_recorded'] = test_dataset['date_recorded'].map(
    lambda x: x.month
)

categorical_vars = train_dataset.select_dtypes(exclude=np.number)

## FEATURE ELIMINATION
variables_to_drop = [
    "scheme_name",
    "recorded_by",
    "region_code",
    'amount_tsh',
    'num_private'
]

for col in categorical_vars.columns:
    if len(train_dataset[col].unique()) > 100:
        variables_to_drop.append(col)

train_dataset.drop(columns=variables_to_drop, inplace=True)
test_dataset.drop(columns=variables_to_drop, inplace=True)

## MISSING VALUES IMPUTATION
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

## IMPUTATION OF ZEROES (MISSING VALUES FOR CONSTRUCTION YEAR)
fill_1 = np.mean(train_dataset.loc[
    (train_dataset['construction_year'] > 0) &
    (train_dataset['status_group'] == "functional"),
    "construction_year"
])

fill_2 = np.mean(train_dataset.loc[
    (train_dataset['construction_year'] > 0) &
    (train_dataset['status_group'] == "non functional"),
    "construction_year"
])

fill_3 = np.mean(train_dataset.loc[
    (train_dataset['construction_year'] > 0) &
    (train_dataset['status_group'] == "functional needs repair"),
    "construction_year"
])

train_dataset.loc[
    (train_dataset['construction_year'] == 0) &
    (train_dataset['status_group'] == "functional"),
    "construction_year"
] = fill_1

train_dataset.loc[
    (train_dataset['construction_year'] == 0) &
    (train_dataset['status_group'] == "non functional"),
    "construction_year"
] = fill_2

train_dataset.loc[
    (train_dataset['construction_year'] == 0) &
    (train_dataset['status_group'] == "functional needs repair"),
    "construction_year"
] = fill_3

test_construction_year = pd.read_csv("../../data/construction_year_test.csv")

test_dataset.loc[
    test_dataset['construction_year'] == 0, 'construction_year'
] = test_construction_year['construction_year']

train_dataset['age'] = train_dataset['year_recorded'] - train_dataset[
    'construction_year'
]

test_dataset['age'] = test_dataset['year_recorded'] - test_dataset[
    'construction_year'
]

train_dataset.to_csv("../train-preprocessed.csv")
test_dataset.to_csv("../test-preprocessed.csv")
