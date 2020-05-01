import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer,SimpleImputer
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv("train.csv",index_col=0)
target = "Survived"

X = df.drop(target,axis=1)
Y = df[target]

X["Pclass"] = X["Pclass"].astype('category')
#grab first letter of cabin and create a category "Missing" for those who don't have a cabin number
X["Cabinletter"] = X["Cabin"].str[0]
X.loc[X["Cabinletter"].isnull(),"Cabinletter"]  = 'Missing_val'
#drop useless columns
X = X.drop(["Name","Cabin","Ticket"],axis=1)

#Create preprocessing pipelines
cat_variables = ["Sex","Cabinletter","Pclass","Embarked"]
quant_columns = ["SibSp","Parch","Fare","Age"]

Transfo_qual = Pipeline(steps=[
    ('impute_qual', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
    ('one_hot_encode_cat', OneHotEncoder(handle_unknown='error',drop='first'))
])
Transfo_quant = Pipeline(steps=[
    ('impute_quant', IterativeImputer(random_state=0)),
    ('scale_quant', RobustScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Transfo_qual, cat_variables),
        ('num', Transfo_quant, quant_columns)
    ])

# Add a RF classifier
clf = Pipeline(steps=[
    ("preprocess",preprocessor),
    ('classifier',RandomForestClassifier(max_depth= 5, min_samples_leaf= 1, n_estimators= 300))
])
clf.fit(X,Y)

# Streamlit interface
st.title("Titanic : likelihood of survival")

Sex = st.sidebar.radio("Passenger gender",df.Sex.unique())
Fare = st.sidebar.slider('Fare',df.Fare.min(),df.Fare.max())
Pclass = st.sidebar.radio("Travel class",df.Pclass.unique())
Age = st.sidebar.slider('Age', df.Age.min(), df.Age.max(),step=1.0)

SibSp = st.sidebar.slider('Siblings',int(df.SibSp.min()),int(df.SibSp.max()))
Parch = st.sidebar.slider('Parch',int(df.Parch.min()),int(df.Parch.max()),step=1)
Embarked = st.sidebar.radio("Embarked at",('Cherbourg', 'Southampton','Queenstown'))
Cabinletter = st.sidebar.radio("Cabin Letter",X["Cabinletter"].unique())

# Reprocess embarkation port full names back to letters
dict_embarked = {"Cherbourg":"C","Queenstown":"Q","Southampton":"S"}    
Embarked = dict_embarked[Embarked]

#Create instance of passenger and predict
instance_df = pd.DataFrame(columns=X.columns)
instance_df.loc[0] = [Pclass,Sex,Age,SibSp,Parch,Fare,Embarked,Cabinletter]
prob_survival = round(clf.predict_proba(instance_df)[0][1]*100,2)
st.subheader("Probability of survival : {} %".format(prob_survival))
