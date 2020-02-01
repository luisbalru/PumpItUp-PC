import numpy as np
import csv
import pandas as pd

from scipy import sparse

import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import KNNImputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

types = {
    "id" : "int",
    "amount_tsh" : "float",
    "date_recorded" : "datetime64",
    "funder" : "str",
    "gps_height" : "float",
    "installer" : "str",
    "longitude" : "float",
    "latitude" : "float",
    "wpt_name" : "str",
    "num_private" : "int",
    "basin" : "str",
    "subvillage" : "str",
    "region" : "str",
    "region_code" : "int",
    "district_code" : "int",
    "lga" : "str",
    "ward" : "str",
    "population" : "int",
    "public_meeting" : "bool",
    "recorded_by" : "str",
    "scheme_management" : "str",
    "scheme_name" : "str",
    "permit" : "bool",
    "construction_year" : "int",
    "extraction_type" : "str",
    "extraction_type_group" : "str",
    "extraction_type_class" : "str",
    "management" : "str",
    "management_group" : "str",
    "payment" : "str",
    "payment_type" : "str",
    "water_quality" : "str",
    "quality_group" : "str",
    "quantity" : "str",
    "quantity_group" : "str",
    "source" : "str",
    "source_type" : "str",
    "source_class" : "str",
    "waterpoint_type" : "str",
    "waterpoint_type_group" : "str"
}

categorical_columns = ["funder",
                        "installer",
                        "wpt_name",
                        "basin",
                        "subvillage",
                        "region",
                        "lga",
                        "ward",
                        "recorded_by",
                        "scheme_management",
                        "scheme_name",
                        "extraction_type",
                        "extraction_type_group",
                        "extraction_type_class",
                        "management",
                        "management_group",
                        "payment",
                        "payment_type",
                        "water_quality",
                        "quality_group",
                        "quantity",
                        "quantity_group",
                        "source",
                        "source_type",
                        "source_class",
                        "waterpoint_type",
                        "waterpoint_type_group"]


def readData(values_train_route="../data/train_values.csv", labels_train_route="../data/train_labels.csv", values_test_route="../data/test_values.csv"):
    labels = None
    d=None
    with open(labels_train_route) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        i=1
        names = []
        d = []
        for row in spamreader:
            if i==1:
                d = {"id": []}
                names.append("id")
                for r in row[1:]:
                    d[r] = []
                    names.append(r)
            else:
                for r,n in zip(row, names):
                    d[n].append(r)
            i+=1
    labels = pd.DataFrame(d)
    labels = np.array(labels["status_group"])

    data_train = None
    with open(values_train_route) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        i=1
        names = []
        d = []
        for row in spamreader:
            if i==1:
                d = {"id": []}
                names.append("id")
                for r in row[1:]:
                    d[r] = []
                    names.append(r)
            else:
                for r,n in zip(row, names):
                    d[n].append(r)
            i+=1

    data_train = pd.DataFrame(d)
    data_train = data_train.astype(types)

    data_test = None
    d=None
    with open(values_test_route) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        i=1
        names = []
        d = []
        for row in spamreader:
            if i==1:
                d = {"id": []}
                names.append("id")
                for r in row[1:]:
                    d[r] = []
                    names.append(r)
            else:
                for r,n in zip(row, names):
                    d[n].append(r)
            i+=1

    data_test = pd.DataFrame(d)
    data_test = data_test.astype(types)

    index_train = list(range(len(data_train)))
    index_test = list(range(len(data_train),len(data_train)+len(data_test)))

    data = pd.concat([data_train, data_test])

    # Convertimos la fecha a float
    min_date = data["date_recorded"].min()
    for i in range(len(data)):
        # In days
        data["date_recorded"].iloc[i] = (data["date_recorded"].iloc[i]-min_date).total_seconds()/86400

    #data = data.drop(columns=["date_recorded"])

    for cat in categorical_columns:
        data[cat] = pd.Categorical(data[cat])

    for cat in categorical_columns:
        print("Filtering the column " + cat + " before One Hot Encoding, num. categories: " + str(len(data[cat].cat.categories)))
        if len(data[cat].value_counts())<100:
            continue

        threshold = data[cat].value_counts().iloc[0]*0.1
        new_cat = "aglomerate"

        bad_categories = np.array(data[cat].value_counts().index[np.where(data[cat].value_counts()<=threshold)[0]].astype("str"))
        data[cat] = data[cat].cat.add_categories(new_cat)

        data[cat].iloc[np.where(data[cat].isin(bad_categories))[0]]=new_cat

        data[cat] = data[cat].cat.remove_unused_categories()

    for cat in categorical_columns:
        data[cat] = pd.Categorical(data[cat].cat.codes)

    data = pd.DataFrame(IterativeImputer().fit_transform(data), columns=data.keys())

    data = pd.get_dummies(data,prefix=categorical_columns, columns=categorical_columns)

    return data.iloc[index_train], labels, data.iloc[index_test]
