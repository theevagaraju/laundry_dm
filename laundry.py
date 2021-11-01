import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
#alt.data_transformers.enable('data_server')
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
#from datetime import datetime
from datetime import datetime as dt

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(rc={'figure.figsize':(11,6)})

from scipy.stats import spearmanr 
import missingno as msno

from boruta import BorutaPy
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook, tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree

st.title("Dashboard: Laundary Dataset")
st.markdown("""
This app is developed by Syed Ishfar Shafiq, Theevagaraju, and Rishidharan to perform an analysis on Laundry Dataset
""")

@st.cache
def load_data(classifier_name):
    df_FS = pd.read_csv('LaundryData.csv')
    del df_FS['No']
    df_FS.dropna(axis=0, subset=['Age_Range'], inplace=True)
    df_FS['Gender'].fillna('male', inplace = True)
    df_FS['Race'].fillna('Other', inplace = True)
    df_FS['Body_Size'].fillna('moderate', inplace = True)
    df_FS['With_Kids'].fillna('no', inplace = True)
    df_FS['Kids_Category'].fillna('Other', inplace = True)
    df_FS['Basket_Size'].fillna('small', inplace = True)
    df_FS['Basket_colour'].fillna('Other', inplace = True)
    df_FS['Shirt_Colour'].fillna('Other', inplace = True)
    df_FS['shirt_type'].fillna('short_sleeve', inplace = True)
    df_FS['Pants_Colour'].fillna('Other', inplace = True)
    df_FS['pants_type'].fillna('short', inplace = True)
    df_FS['Wash_Item'].fillna('clothes', inplace = True)
    df_FS['Attire'].fillna('Other', inplace = True)
    df_FS['Age_Range'] = df_FS['Age_Range'].apply(np.int64)

    return df_FS 


classifier_name = st.sidebar.selectbox("Select Classifier",("Naive Bayes","K-Nearest Neighbors","Random Forest"))

df = load_data(classifier_name)
df_fs = df.copy()
del df_fs['Time']
del df_fs['Date']
st.write(df_fs.head(5))

 

st.write("Age groups are most use the laundary")
fig = plt.figure()
analysis = df_fs.copy()
analysis['AGE_GROUP'] = pd.cut(analysis['Age_Range'], bins=[0,10,20,30,40,50,60,70,80,90,100])
analysis['AGE_GROUP'].value_counts().plot(kind='bar')
st.pyplot(fig)

st.write("Number of customer are with kids")
fig = plt.figure()
analysis['With_Kids'].value_counts().plot(kind='bar')
st.pyplot(fig)

fig = plt.figure()
sns.distplot(df_fs["Age_Range"], bins=10)
st.write("Skewness",df_fs["Age_Range"].skew()) 
st.pyplot(fig)

fig = plt.figure()
b=sns.countplot(x='Washer_No', data = df_fs)
plt.title('Number of Washer ')
st.pyplot(fig)

fig = plt.figure(figsize=(7,7))
corr = df_fs.corr()
sns.heatmap(corr, vmax=.8, square=True, annot=True, fmt='.2f',
           annot_kws={'size': 15}, cmap=sns.color_palette("Reds"))
st.pyplot(fig)

categorical = ['Race','Gender','Body_Size', 'With_Kids','Kids_Category','Basket_Size','Basket_colour','Attire','Shirt_Colour','shirt_type','Pants_Colour','pants_type','Wash_Item','Spectacles']
d = defaultdict(LabelEncoder) 
df_fs[categorical] = df_fs[categorical].apply(lambda x: d[x.name].fit_transform(x.astype(str)))

class_variable = st.sidebar.selectbox("Class Variables",("Gender","Spectacles"))
if(class_variable == "Gender"):
   X = df_fs.drop('Gender', axis=1) 
   y = df_fs['Gender']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=10)
elif (class_variable == "Spectacles"): 
   X = df_fs.drop('Spectacles', axis=1) 
   y = df_fs['Spectacles']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=10)
else:
   X = df_fs.drop('Gender', axis=1) 
   y = df_fs['Gender']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=10)


if(classifier_name == "K-Nearest Neighbors"):
   st.title(classifier_name)
   
   #Classification-KNN
   k=11
   knn = KNeighborsClassifier(n_neighbors=k)
   knn.fit(X_train,y_train)
   st.write('KNN Score= {:.2f}'.format(knn.score(X_test, y_test)))

   y_pred = knn.predict(X_test)
   confussion_m = confusion_matrix(y_true=y_test, y_pred= y_pred)

   st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred, pos_label=0)))
   st.write('Recall= {:.2f}'.format(recall_score(y_test, y_pred, pos_label=0)))
   st.write('F1= {:.2f}'.format(f1_score(y_test, y_pred, pos_label=0)))
   st.write('Accuracy= {:.2f}'.format(accuracy_score(y_test, y_pred)))

   st.title("Class Variable:" + class_variable)

   #plot knn
   prob_KNN = knn.predict_proba(X_test)
   prob_KNN = prob_KNN[:, 1]
   fpr_knn, tpr_knn, thresholds_DT = roc_curve(y_test, prob_KNN) 

   fig = plt.figure()
   plt.plot(fpr_knn, tpr_knn, color='red', label='KNN') 
   plt.plot([0, 1], [0, 1], color='green', linestyle='--')
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('K-Nearest Neighbors (KNN) Curve')
   plt.legend()

   st.pyplot(fig)

elif(classifier_name == "Naive Bayes"):
   st.title(classifier_name)

   #Classification-NB
   nb = GaussianNB()
   nb.fit(X_train, y_train)
   y_pred = nb.predict(X_test)

   confussion_m = confusion_matrix(y_true=y_test, y_pred= y_pred)
   st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred, pos_label=0)))
   st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred, pos_label=0)))
   st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred, pos_label=0)))
   st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))
   st.title("Class Variable:" + class_variable)
   #plot NB
   prob_NB = nb.predict_proba(X_test)
   prob_NB = prob_NB[:, 1]
   fpr_NB, tpr_NB, thresholds_DT = roc_curve(y_test, prob_NB) 

   fig = plt.figure()
   plt.plot(fpr_NB, tpr_NB, color='orange', label='NB')
   plt.plot([0, 1], [0, 1], color='green', linestyle='--')
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Naive Bayes (NB) Curve')
   plt.legend()
   st.pyplot(fig)

else:
   st.title(classifier_name)

   rf = RandomForestClassifier(random_state=10)
   rf.fit(X_train, y_train)
   y_pred = rf.predict(X_test)
   confusion_majority=confusion_matrix(y_test, y_pred)

   confussion_m = confusion_matrix(y_true=y_test, y_pred= y_pred)

   st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred, pos_label=0)))
   st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred, pos_label=0)))
   st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred, pos_label=0)))
   st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))
   st.title("Class Variable:" + class_variable)
   #plot RF
   fig = plt.figure()
   prob_RF = rf.predict_proba(X_test)
   prob_RF = prob_RF[:, 1]
   fpr_rf, tpr_rf, thresholds_DT = roc_curve(y_test, prob_RF)
   plt.plot(fpr_rf, tpr_rf, color='blue', label='RF') 
   plt.plot([0, 1], [0, 1], color='green', linestyle='--')
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Receiver Operating Characteristic (ROC) Curve')
   plt.legend()

   st.pyplot(fig)


   
