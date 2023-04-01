#importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
import joblib 

#loading dataset
df = pd.read_csv("data\mushrooms.csv")

#encoding categorical data into numerical data
encoder = OrdinalEncoder()
df = encoder.fit_transform(df)

# dividing dataset into X & y
X = df[:,1:]
y = df[:,0]

#splitting data into X_train, X_test, y_train, y_test  
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

# creating model
KNN_model = KNeighborsClassifier()

file_name = "models/model.joblib"
# fitting dataset into model and dumping it into KNN_model
joblib.dump(KNN_model.fit(X_train, y_train), filename=file_name)
saved_model = KNN_model.fit(X_train, y_train)

#accuracy score of model
KNN_model.score(X_test, y_test)