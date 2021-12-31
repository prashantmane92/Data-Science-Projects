import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import re
import pickle

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,MinMaxScaler,StandardScaler
from scipy.stats import shapiro, normaltest, kstest, zscore
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn import set_config
set_config(display="diagram")

df = pd.read_csv('Pune_House_Data.csv')

df.drop("society",axis=1,inplace=True)

df[["First","Second"]] =df["size"].str.split(expand=True)

df.drop("size",axis=1,inplace=True)

def clean(s):
    pattern = re.compile(r"[\d.]+")
    try:
        s = float(s)
    except (ValueError):
        if s.find("-") !=-1:
            s=np.mean([float(i) for i in s.split("-")])
        elif s.endswith('Sq. Meter'):
            s = float(pattern.search(s)[0])*10.7639
        elif s.endswith('Perch'):
            s = float(pattern.search(s)[0])*0.00367309
        elif s.endswith('Sq. Yards'):
            s = float(pattern.search(s)[0])*9
        elif s.endswith('Guntha'):
            s = float(pattern.search(s)[0])*1089
        elif s.endswith('Acres'):
            s = float(pattern.search(s)[0])*43560
        elif s.endswith('ents'):
            s = float(pattern.search(s)[0])*435.56
        elif s.endswith('Grounds'):
            s = float(pattern.search(s)[0]) *2400 
    return s

df["total_sqft"] = df["total_sqft"].apply(clean).astype('int')

df['availability'] = df['availability'].apply(lambda x: x + '-2022')

current_date =datetime.datetime(2022, 1, 1).date()
df['availability'].replace({'Ready To Move-2022':'1-Jan-2022','Immediate Possession-2022':'1-Jan-2022'},inplace=True)

df["availability"] =df["availability"].apply(lambda x :abs((datetime.datetime.strptime(x,"%d-%b-%Y").date())-current_date).days)

df["availability"] = df["availability"]/30

outlier_sqft = df[abs(zscore(df["total_sqft"]))>3].index
df.drop(index = outlier_sqft,inplace=True)

df.drop("Second",axis=1,inplace=True)

df = df.rename(columns={'First':'Bedrooms'})

df["Bedrooms"] = df["Bedrooms"].astype('float')

df = df[["total_sqft","availability","bath","balcony","Bedrooms","area_type","site_location","price"]]
df

X = df.drop('price',axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.head(1)

MVT = ColumnTransformer([("Median_Fillna",SimpleImputer(strategy='median'),[0,1]),
                                      ("Mode1_Fillna",SimpleImputer(strategy='most_frequent'),[2,3]),
                                      ("0_Fillna",SimpleImputer(strategy = "constant",fill_value=0),[4]),
                                      ("Mode2_Fillna",SimpleImputer(strategy='most_frequent'),[5,6])],remainder='passthrough')

ST = ColumnTransformer([("SC",StandardScaler(),[0,1,2,3,4])],remainder='passthrough')

OHT = ColumnTransformer([("AT_OHE",OneHotEncoder(sparse=False,drop='first',handle_unknown = 'ignore'),[5,6])], remainder='passthrough')

pipe_lr = Pipeline([('Missing_Value', MVT), ('Scaling', ST), ("OHE", OHT), ("LR",LinearRegression())])
pipe_lr.fit(X_train,y_train)
y_pred_test = pipe_lr.predict(X_test)
y_pred_train = pipe_lr.predict(X_train)

pickle.dump(pipe_lr,open("hpp_lr_pipe.pkl",'wb'))