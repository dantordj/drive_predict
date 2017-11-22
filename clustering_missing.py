import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import numpy as np


def cluster_missing(X_train):

	if X_train.shape[1] == 59:
		X = X_train.drop(["id", "target"], axis = 1)

	else:
		X = X_train.drop(["id"], axis = 1)


	na_count = X.isnull().sum()
	na_columns = list(na_count[na_count>0].index.values)

	print("columns with missing values:")
	print(na_columns)



	#create df only with columns with no missing values
	X_no_missing = X.drop(na_columns, axis = 1)
 
	#one hot encoding of categorical features
	cat_columns_no_missing = list(filter(lambda x: x.endswith("cat"),
                                     X_no_missing.columns.values))
	X_no_missing_oh = pd.get_dummies(X_no_missing, columns = cat_columns_no_missing)   

	#train kmeans
	kmeans = MiniBatchKMeans(n_clusters = 15, random_state = 0, batch_size = 2000)
	kmeans.fit(X_no_missing_oh)
	print("Clustersize: \n")
	print(pd.Series(kmeans.labels_).value_counts())

	#store cluster labels in df
	X["cluster"] = kmeans.labels_

	#for columns with missing values, drop missing values and find median or most common value - per cluster
	Values_replace_missing = pd.DataFrame()

	for i in na_columns:
	    clean_df = X[["cluster", i]].dropna()
	    if i.endswith("cat"):
	        Values_replace_missing[i] = clean_df.groupby(["cluster"]).agg(lambda x:x.value_counts().index.values[0])
	    else:
	        Values_replace_missing[i] = clean_df.groupby(["cluster"]).median() 

	print(Values_replace_missing)

	#replace missing values with median or most common value in the same cluster
	for cl, cat in ((x, y) for x in range(15) for y in na_columns):
	    X_train.loc[(X["cluster"] == cl) & pd.isnull(X[cat]), cat] = Values_replace_missing.loc[cl, cat]

	#print remaining missing values (should be zero)
	print("\n remaining missing values: " + str(X_train.isnull().sum().sum()))

	return X_train