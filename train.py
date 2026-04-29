from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn
import pandas as pd
import numpy as np
import pickle

mlflow.set_tracking_uri("file:./mlruns")

df = pd.read_csv("data/data.csv")

X = df[["hours"]]
Y = df["pass"]

x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.2,random_state=42)

with mlflow.start_run():
    
    model = RandomForestClassifier(n_estimators=100)
    
    model.fit(x_train,y_train)
    
    accuracy = model.score(x_test,y_test)
    
    mlflow.log_param("n_estimators",100)
    mlflow.log_metric("Accuracy",accuracy)
    mlflow.sklearn.log_model(model,name="model")
    
    print("Accuracy",accuracy)




