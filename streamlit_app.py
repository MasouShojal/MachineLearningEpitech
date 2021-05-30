import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import metrics 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
import ipywidgets as widgets

from datetime import datetime as dt
from urllib.error import URLError

st.title("Determine how healthy you are !")

df = pd.read_csv('heart.csv',delimiter=',')
df2 = pd.read_csv('heart.csv',delimiter=',')
df['output'] = df['output'].map({0: "high", 1: "low"})
features = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'caa', 'thall']
target = ['output']
X = df[features]
Y = df[target]
output_names = df['output'].unique()
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.20)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# convert labels to numerical values 
Y_train = LabelEncoder().fit_transform(np.asarray(Y_train).ravel())
Y_test = LabelEncoder().fit_transform(np.asarray(Y_test).ravel())
model = RandomForestClassifier()
model.fit(X_train,Y_train)
# fit a smaller forest with a maximum depth of 3 (this is how many consecutive 
# decision the algorithm can make). As a consequence, the accuracy will be lower
# but it'll be easier to visualise it
small_rf = RandomForestClassifier(max_depth=5)
# fit the forest to the training data 
small_rf.fit(X_train,Y_train)
# get predictions on the test data 
Y_pred=small_rf.predict(X_test)
trees = small_rf.estimators_
tree.plot_tree(trees[0],
            feature_names = features, 
            class_names = output_names,
            filled = True);



def get_predictions(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12):
    feature = scaler.transform(np.asarray([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12]).reshape((1,-1)))
    quality = output_names[model.predict(feature).item()]
    st.write('the risk of this person to develop an heart disease is {}. '.format(quality))


def prep_dataset(age, gender_name, chest_pain_name, resting_blood_pressure, cholesterol, fast_blood_sugar, rest_electro_name, max_heart_rate, exercice_angina, oldpeak, major_vessels, thal):
    ##Change gender name to numerical value
    if gender_name == "Female":
        gender = 1
    else:
        gender = 0
    ##Change chest_pain_name to numerical value
    if chest_pain_name == "Typical angina":
        chest_pain = 0
    elif chest_pain_name == "Atypical angina":
        chest_pain = 1
    elif chest_pain_name == "Non-anginal pain":
        chest_pain = 2
    else:
        chest_pain = 3
    ##Change fast_blood_sugar to numerical value
    if fast_blood_sugar == "False":
        fast_blood = 0
    else:
        fast_blood = 1
    ##Change rest_electro to numerical value
    if rest_electro_name == "Normal":
        rest_electro = 0
    elif rest_electro_name == "Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)":
        rest_electro = 1
    else:
        rest_electro = 2
    ##Change exercice_angina to numerical value
    if exercice_angina == "No":
        exe_angina = 0
    else:
        exe_angina = 1
    ##Display values
    #st.write("age = ", age, "gender = ", gender, "chest pain = ", chest_pain, "resting blood pressure = ", resting_blood_pressure, "cholesterol = ", cholesterol, "fast blood sugar = ",fast_blood_sugar, "rest electro = ", rest_electro, "max heart rate = ",max_heart_rate, "exercice angine = ", exe_angina,"oldpeak = ", oldpeak, "major vessels = ",major_vessels, "thal = ",thal)
    get_predictions(age, gender, chest_pain, resting_blood_pressure, cholesterol, fast_blood, rest_electro, max_heart_rate, exe_angina, oldpeak, major_vessels, thal)


def complete_form():
    with st.form("my_form"):
        age = st.slider('How old are you ?',  0, 77, 25)
        gender_name = st.selectbox("What are your gender ?", ("Female", "Male"))
        chest_pain_name = st.selectbox("What type of chest pain do you have ?", ("Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic")) #change gender_name to numerical value with 0,1,2,3
        resting_blood_pressure = st.slider('What is your resting blood pressure ?',  94, 200, 125)
        cholesterol = st.slider('What is your cholesterol rate ?', 126, 564, 180)
        fast_blood_sugar = st.selectbox("Is your fasting blood glucose above 120mg/dl ?", ("False", "True"))
        rest_electro_name = st.selectbox("What is your resting electrocardiographic results ?", ("Normal", "Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)","Showing probable or definite left ventricular hypertrophy by Estes' criteria"))
        max_heart_rate = st.slider("What is the maximum heart rate you achieved ?", 71 , 202, 125)
        exercice_angina = st.selectbox("Does exercising cause angina?", ("No", "Yes"))
        oldpeak = st.slider("What was your previous peak?", 0.00, 6.20, 3.00)
        major_vessels = st.slider("What is your number of major vessels ?", 0,4,2)
        thal = st.slider("How many times have you done thal ?",0,3,2)

    
        submitted = st.form_submit_button("Submit")
        if submitted:
            prep_dataset(age, gender_name, chest_pain_name, resting_blood_pressure, cholesterol, fast_blood_sugar, rest_electro_name, max_heart_rate, exercice_angina, oldpeak, major_vessels, thal)


def main():
    complete_form()


if __name__ == '__main__':
	main()