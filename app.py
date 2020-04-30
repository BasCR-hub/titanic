import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler

df_preprocessed = pd.read_csv("train.csv",index_col=0)

#change class type to category
df_preprocessed["Pclass"] = df_preprocessed["Pclass"].astype('category')

#grab first letter of cabin and create a category "Missing" for those who don't have a cabin number
df_preprocessed["Cabinletter"] = df_preprocessed["Cabin"].str[0]
df_preprocessed.loc[df_preprocessed["Cabinletter"].isnull(),"Cabinletter"]  = 'Missing'

#drop useless columns
df_preprocessed = df_preprocessed.drop(["Name","Cabin","Ticket"],axis=1)

#onehotencode categorical data
df_preprocessed = pd.get_dummies(df_preprocessed, drop_first=True)

#impute missing values
imp_mean = IterativeImputer(random_state=0)
df_preprocessed = pd.DataFrame(imp_mean.fit_transform(df_preprocessed),columns = df_preprocessed.columns)

#robust scale numerical data
rs_scaler = RobustScaler()
df_preprocessed["Age"] = rs_scaler.fit_transform(df_preprocessed.Age.values.reshape(-1,1))
df_preprocessed["Fare"] = rs_scaler.fit_transform(df_preprocessed.Fare.values.reshape(-1,1))
df_preprocessed["SibSp"] = rs_scaler.fit_transform(df_preprocessed.SibSp.values.reshape(-1,1))
df_preprocessed["Parch"] = rs_scaler.fit_transform(df_preprocessed.Parch.values.reshape(-1,1))

## train predictive model using preprocessed data
X = df_preprocessed.loc[:, df_preprocessed.columns != "Survived"]
Y = df_preprocessed["Survived"]
rf = RandomForestClassifier(max_depth= 10, min_samples_leaf= 1, n_estimators= 300)
rf.fit(X,Y)

## load minimally preprocessed dataframe to display user-friendly variables
df_min_preprocessed = pd.read_csv("df_min_preprocessed.csv",index_col = 0)

# @st.cache
# def get_data():
#     df = pd.read_csv("preprocessed_titanic.csv",index_col=0)
#     return df
# df = get_data()
# X = df.loc[:, df.columns != "Survived"]
# Y = df["Survived"]
# rf = RandomForestClassifier(max_depth= 10, min_samples_leaf= 1, n_estimators= 300)
# rf.fit(X,Y)

st.title("Titanic : likelihood of survival")

gender = st.sidebar.radio("Passenger gender",('Male', 'Female'))
Fare = st.sidebar.slider('Fare',df_min_preprocessed.Fare.min(),df_min_preprocessed.Fare.max())
travel_class = st.sidebar.radio("Travel class",('1st', '2nd','3rd'))
Age = st.sidebar.slider('Age', df_min_preprocessed.Age.min(), df_min_preprocessed.Age.max(),step=1.0)

SibSp = st.sidebar.slider('Siblings',int(df_min_preprocessed.SibSp.min()),int(df_min_preprocessed.SibSp.max()))
Parch = st.sidebar.slider('Parch',int(df_min_preprocessed.Parch.min()),int(df_min_preprocessed.Parch.max()),step=1)
embarked_at = st.sidebar.radio("Embarked at",('Cherbourg', 'Southampton','Queenstown'))
cabin_letter = st.sidebar.radio("Cabin Letter",df_min_preprocessed["Cabinletter"].unique())

## Handle one hot encoded categorical variables (by hand)
Pclass_2, Pclass_3 = (0,0)
if travel_class == "2nd":
    Pclass_2 = 1
elif travel_class == "3rd":
    Pclass_3 = 1

Sex_male = 0
if gender == "Male":
    Sex_male = 1

Embarked_Q, Embarked_S = (0,0)
if embarked_at == 'Southampton':
    Embarked_S = 1
elif embarked_at == 'Queenstown':
    Embarked_Q = 1

Cabinletter_B,Cabinletter_C,Cabinletter_D,Cabinletter_E,Cabinletter_F,Cabinletter_G,Cabinletter_T = (0,0,0,0,0,0,0)
if cabin_letter == "B":
    Cabinletter_B = 1
elif cabin_letter == "C":
    Cabinletter_C = 1
elif cabin_letter == "D":
    Cabinletter_D = 1
elif cabin_letter == "E":
    Cabinletter_E = 1
elif cabin_letter == "F":
    Cabinletter_F = 1
elif cabin_letter == "G":
    Cabinletter_G = 1
elif cabin_letter == "T":
    Cabinletter_H = 1


features = np.array([Age,SibSp,Parch,Fare,Pclass_2,Pclass_3,Sex_male,Embarked_Q,Embarked_S,Cabinletter_B,Cabinletter_C,Cabinletter_D,Cabinletter_E,Cabinletter_F,Cabinletter_G,Cabinletter_T])
survival_prob = round(rf.predict_proba(features.reshape(1,-1))[0][1]*100,2)
st.subheader("{} %".format(survival_prob))
