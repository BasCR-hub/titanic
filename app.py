import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

# df = pd.read_csv("preprocessed_titanic.csv",index_col=0)
# X = df.loc[:, df.columns != "Survived"]
# Y = df["Survived"]
# rf = RandomForestClassifier(max_depth= 10, min_samples_leaf= 1, n_estimators= 300)
# rf.fit(X,Y)



@st.cache
def get_data():
    df = pd.read_csv("preprocessed_titanic.csv",index_col=0)
    return df
df = get_data()
X = df.loc[:, df.columns != "Survived"]
Y = df["Survived"]
rf = RandomForestClassifier(max_depth= 10, min_samples_leaf= 1, n_estimators= 300)
rf.fit(X,Y)

st.title("Titanic : likelihood of survival")

gender = st.sidebar.radio("Passenger gender",('Male', 'Female'))
Fare = st.sidebar.slider('Fare',df.Fare.min(),df.Fare.max())
travel_class = st.sidebar.radio("Travel class",('1st', '2nd','3rd'))
Age = st.sidebar.slider('Age', 0, 100, 25)

SibSp = st.sidebar.slider('Siblings',df.SibSp.min(),df.SibSp.max(),step=1.0)
Parch = st.sidebar.slider('Parch',df.Parch.min(),df.Parch.max(),step=1.0)
embarked_at = st.sidebar.radio("Embarked at",('Cherbourg', 'Southampton','Queenstown'))
cabin_letter = st.sidebar.radio("Cabin Letter",('B', 'C','D','E','F','G','T','No information'))

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
survival_prob = round(rf.predict_proba(features.reshape(1,16))[0][1]*100,2)
st.subheader("{} %".format(survival_prob))
