import numpy as np
import csv
import pandas as pd

from scipy import sparse

import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer

#from fancyimpute import KNN

from sklearn.impute import KNNImputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

types = {
    "id" : "int",
    #Paco"amount_tsh" : "float",
    "date_recorded" : "datetime64",
    #"funder" : "str",
    "gps_height" : "float",
    #"installer" : "str",
    "longitude" : "float",
    "latitude" : "float",
    #"wpt_name" : "str",
    ##"num_private" : "int",
    "basin" : "str",
    #"subvillage" : "str",
    "region" : "str",
    #Paco"region_code" : "int",
    "district_code" : "int",
    #Paco"lga" : "str",
    #"ward" : "str",
    ##"population" : "int",
    "public_meeting" : "bool",
    ##"recorded_by" : "str",
    "scheme_management" : "str",
    #Paco"scheme_name" : "str",
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

categorical_columns = [#"funder",
                        #"installer",
                        #"wpt_name",
                        "basin",
                        #"subvillage",
                        "region",
                        #Paco"lga",
                        #"ward",
                        ##"recorded_by",
                        "scheme_management",
                        #Paco"scheme_name",
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
                        #$"region_code",
                        #$"district_code"]

numerical_column = [#Paco"amount_tsh",
                    "gps_height",
                    "longitude",
                    "latitude"]


def readData(values_train_route="../data/train_values.csv", labels_train_route="../data/train_labels.csv", values_test_route="../data/test_values.csv", labels_test_route = "../data/test_labels_xgboost.csv"):
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

    labels_test = pd.read_csv(labels_test_route)
    labels_test = np.array(labels_test["status_group"])

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

    #data = data.drop(columns=["wpt_name", "installer", "funder", "ward", "num_private", "recorded_by", "subvillage"])
    data = data.drop(columns=['scheme_name', 'recorded_by', 'region_code', 'amount_tsh', 'num_private', 'funder', 'installer', 'wpt_name', 'subvillage', 'lga', 'ward'])

    # Pasamos los datos perdidos a nan
    data["gps_height"][data["gps_height"]==0]=np.nan
    data["longitude"][data["longitude"]==0]=np.nan
    data["latitude"][data["latitude"]==-2e-8]=np.nan
    data["population"][data["population"]==0]=np.nan
    data["construction_year"][data["construction_year"]==0]=np.nan

    print("Marcando outliers univariantes como anomalias...")
    for name in  numerical_column:
        clean = np.array(data[name])
        clean = clean[clean!=np.nan]
        mean = np.mean(clean)
        median = np.median(clean)
        std = np.std(clean)
        for i in range(len(data)):
            if data[name].iloc[i]!=np.nan:
                if data[name].iloc[i]>mean+5*std or data[name].iloc[i]<mean-5*std:
                    data[name].iloc[i] = median


    ind_functional = np.where(labels=="functional")[0]
    functional_train = data.iloc[index_train]["construction_year"].iloc[ind_functional]
    mean_functional_train = np.mean(functional_train[functional_train!=np.nan])

    ind_non_functional = np.where(labels=="non functional")[0]
    non_functional_train = data.iloc[index_train]["construction_year"].iloc[ind_non_functional]
    mean_non_functional_train = np.mean(non_functional_train[non_functional_train!=np.nan])

    ind_need_repair = np.where(labels=="functional needs repair")[0]
    need_repair_train = data.iloc[index_train]["construction_year"].iloc[ind_need_repair]
    mean_need_repair_train = np.mean(need_repair_train[need_repair_train!=np.nan])

    for i in range(len(index_train)):
        if data["construction_year"].iloc[index_train[i]]==np.nan:
            if labels[i]=="functional":
                data["construction_year"].iloc[index_train[i]]=mean_functional_train
            elif labels[i]=="non functional":
                data["construction_year"].iloc[index_train[i]]=mean_non_functional_train
            else:
                data["construction_year"].iloc[index_train[i]]=mean_need_repair_train

    ind_functional = np.where(labels_test=="functional")[0]
    functional_test = data.iloc[index_test]["construction_year"].iloc[ind_functional]
    mean_functional_test = np.mean(functional_test[functional_test!=np.nan])

    ind_non_functional = np.where(labels_test=="non functional")[0]
    non_functional_test = data.iloc[index_test]["construction_year"].iloc[ind_non_functional]
    mean_non_functional_test = np.mean(non_functional_test[non_functional_test!=np.nan])

    ind_need_repair = np.where(labels_test=="functional needs repair")[0]
    need_repair_test = data.iloc[index_test]["construction_year"].iloc[ind_need_repair]
    mean_need_repair_test = np.mean(need_repair_test[need_repair_test!=np.nan])

    for i in range(len(index_test)):
        if data["construction_year"].iloc[index_test[i]]==np.nan:
            if labels_test[i]=="functional":
                data["construction_year"].iloc[index_test[i]]=mean_functional_test
            elif labels_test[i]=="non functional":
                data["construction_year"].iloc[index_test[i]]=mean_non_functional_test
            else:
                data["construction_year"].iloc[index_test[i]]=mean_need_repair_test

    # Convertimos la fecha a float
    for i in range(len(data)):
        # In years
        data["date_recorded"].iloc[i] = data["date_recorded"].iloc[i].year

    data["pump_age"] = data["date_recorded"]-data["construction_year"]
    data = data.drop(columns=["date_recorded", "construction_year"])

    for cat in categorical_columns:
        data[cat] = pd.Categorical(data[cat])

    for cat in categorical_columns:
        print("Filtering the column " + cat + " before One Hot Encoding, num. categories: " + str(len(data[cat].cat.categories)))
        if len(data[cat].value_counts())<100:
            continue

        threshold = data[cat].value_counts().iloc[0]*0.1
        threshold = 1
        new_cat = "aglomerate"

        bad_categories = np.array(data[cat].value_counts().index[np.where(data[cat].value_counts()<=threshold)[0]].astype("str"))
        data[cat] = data[cat].cat.add_categories(new_cat)

        data[cat].iloc[np.where(data[cat].isin(bad_categories))[0]]=new_cat

        data[cat] = data[cat].cat.remove_unused_categories()

    for cat in categorical_columns:
        data[cat] = pd.Categorical(data[cat].cat.codes)

    #data["date_recorded"] = data["date_recorded"].transform(lambda x: np.log(x + 1))
    #data["gps_height"] = data["gps_height"].transform(lambda x: np.log(x + 1))
    #data["latitude"] = data["latitude"].transform(lambda x: np.log(x + 1))
    #data["longitude"] = data["longitude"].transform(lambda x: np.log(x + 1))

    print("Imputando datos...")
    data = pd.DataFrame(IterativeImputer().fit_transform(data), columns=data.keys())

    data = pd.get_dummies(data,prefix=categorical_columns, columns=categorical_columns)

    return data.iloc[index_train], labels, data.iloc[index_test]
