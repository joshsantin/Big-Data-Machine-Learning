#!/usr/bin/env python
# coding: utf-8

# ## Heart Disease Prediction

# Using machine learning to predict heart disease, and later on creating a system for healthcare professionals who can update the patient's parameters. The model will run again on the updated patient's data and alert if there is a risk of heart disease for the new updated values.
# 
# This is unique because a patient may do tests for some other health issue, but if their updated parameters indicate heart disease risk, this system will identify it, that is, it will be identified early rather than it be going unnoticed.
# 
# And the notification will be sent to the Family Doctor as well as the hospital staff, it will look like :-
# 
# <div> <img src="image.jpg" alt="Drawing" style="height: 400px;"/></div>

# In[ ]:


#Importing Libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mysql.connector
import os
from sklearn.preprocessing import MinMaxScaler
import csv
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score

from sklearn.metrics import roc_curve
import sklearn.metrics as metrics

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


#Read Dataset
data_original = pd.read_csv('heart.csv')
df = data_original.copy()


# In[ ]:


#Overview of Data
df


# In[ ]:


#Summary Statistics of Data
df.describe()


# ### Data Cleaning

# In[ ]:


#Checking for null data

df.isna().sum()


# In[ ]:


#Checking for duplicate data
df.duplicated().sum()


# In[ ]:


#Checking for outliers using Boxplots
count = 1

columns_text = ['Age','Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope', 'HeartDisease', 'FastingBS']

for column in df.drop(columns_text, axis = 1):
    fig = plt.figure(figsize = (6,4))
    sns.boxplot(x = df[column], hue = df.HeartDisease, color = 'orange')
    if count == 1: 
        count += 1
    else:
        count == 1


# We see some outliers, we need to remove them. We will use the IQR (Inter Quartile Range)
# 
# IQR = Q3 - Q1

# In[ ]:


#We will use Quantile based flooring and capping with 10% percentile as flooring and 90% percentile as capping

outlier_columns = ['RestingBP','Oldpeak', 'MaxHR', 'Cholesterol']

Q1 = df.quantile(0.1)
Q3 = df.quantile(0.9)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


# In[ ]:


(df == 0).sum(axis = 0)


# In[ ]:


# We will delete all the data with outliers because replacing them will result in inaccurate calculations, 
#and we can't take that risk in health scenario

for x in outlier_columns:
    df = df[(df[x] > lower_bound[x]) & (df[x] < upper_bound[x])]


# In[ ]:


#Replacing Categorical columns

# Select categorical variables

categ = df.select_dtypes(include=object).columns

# One hot encoding
df = pd.get_dummies(df, columns=categ, drop_first= False)  
#df.head()


# In[ ]:


df


# In[ ]:


df = df.drop(columns = ['Sex_F', 'ChestPainType_ASY', 'RestingECG_Normal','ExerciseAngina_N', 'ST_Slope_Down'])


# In[ ]:


df.head()


# In[ ]:


corr_matrix = df.corr(method ='pearson')
plt.figure(figsize=(19,19))
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm')


# In[ ]:


print(corr_matrix["HeartDisease"].sort_values(ascending=False))


# ## Machine Learning

# In[ ]:


# Splitting Data into Training and Test Dataset

from sklearn.model_selection import train_test_split

y = df['HeartDisease']
X = df.drop(columns = 'HeartDisease')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

#We need to scale dataset because some machine learning models like KNN, Linear regression, (anyone with distance models)
#require the data to be normalized

scaler = MinMaxScaler()
model=scaler.fit(X_train)

X_train_scaled = pd.DataFrame(model.transform(X_train))
X_test_scaled = pd.DataFrame(model.transform(X_test))


# In[ ]:


#Confusion Matrix Function

def plot_confusion(data):
    ax = sns.heatmap(data/np.sum(data), annot=True, 
            fmt='.2%', cmap='Blues')

    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()


# ### Naive Bayes Classifier

# In[ ]:


from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred  =  classifier.predict(X_test)
print('Accuracy score: ', round(accuracy_score(y_test, y_pred), 4))

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)

print('Accuracy score: ', round(accuracy_score(y_test, y_pred), 4))
print('F1 Score: ', round(f1_score(y_test, y_pred), 4))


#create ROC curve
plt.plot(fpr,tpr)
plt.title('ROC')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print('roc_auc_score: ', roc_auc_score(y_test, y_pred))

plot_confusion(confusion_matrix(y_test, y_pred))


# ### Logistic Regression

# In[ ]:


# Standard logistic regression
lr = LogisticRegression(solver='liblinear').fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)


fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)

print('Accuracy score: ', round(accuracy_score(y_test, y_pred), 4))
print('F1 Score: ', round(f1_score(y_test, y_pred), 4))


#create ROC curve
plt.plot(fpr,tpr)
plt.title('ROC')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print('roc_auc_score: ', roc_auc_score(y_test, y_pred))

plot_confusion(confusion_matrix(y_test, y_pred))


# ### K-Nearest Neighbour

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

# First model
knn = KNeighborsClassifier(n_neighbors=1)
knn = knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)


fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)

print('Accuracy score: ', round(accuracy_score(y_test, y_pred), 4))
print('F1 Score: ', round(f1_score(y_test, y_pred), 4))


#create ROC curve
plt.plot(fpr,tpr)
plt.title('ROC')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print('roc_auc_score: ', roc_auc_score(y_test, y_pred))
plot_confusion(confusion_matrix(y_test, y_pred))


# ### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt = dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)


fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)

print('Accuracy score: ', round(accuracy_score(y_test, y_pred), 4))
print('F1 Score: ', round(f1_score(y_test, y_pred), 4))

#create ROC curve
plt.plot(fpr,tpr)
plt.title('ROC')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print('roc_auc_score: ', roc_auc_score(y_test, y_pred))
plot_confusion(confusion_matrix(y_test, y_pred))


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# First model
RF = RandomForestClassifier(random_state=200, n_estimators=500)
                            
RF = RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)

print('Accuracy score: ', round(accuracy_score(y_test, y_pred), 4))
print('F1 Score: ', round(f1_score(y_test, y_pred), 4))

#create ROC curve
plt.plot(fpr,tpr)
plt.title('ROC')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print('roc_auc_score: ', roc_auc_score(y_test, y_pred))
plot_confusion(confusion_matrix(y_test, y_pred))


# ### Neural Networks

# In[ ]:


from tensorflow import keras #keras
import tensorflow as tf #tensorflow

from tensorflow.keras import layers, models #neural network architecture
from tensorflow.keras.metrics import BinaryAccuracy #model evaluation
from tensorflow.keras.callbacks import EarlyStopping #regularization
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras import regularizers
from tensorflow.keras.optimizers import Adam


# In[ ]:


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=15, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(8, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))
    model.add(Flatten())
    
    # compile model
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

nn = create_model()

print(nn.summary())


# <div> <img src="0001.jpg" alt="Drawing" style="height: 400px;"/></div>

# In[ ]:


history=nn.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test),epochs=100, batch_size=10)


# In[ ]:


y_pred = np.argmax(nn.predict(X_test), axis=1)
test_acc = accuracy_score(y_test, y_pred)
print(test_acc)

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
plt.plot(fpr,tpr)
plt.title('ROC')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

plot_confusion(confusion_matrix(y_test, y_pred))


# # Creating the system

# ## SQL backend database design
# 
# <div> <img src="relation.png" alt="Drawing" style="height: 400px;"/></div>

# ## Creating notification system and connecting mysql

# In[ ]:


connection = mysql.connector.connect(host = 'localhost',
                              user = 'root',
                              passwd = '',
                              database = 'hospital',
                              use_pure = True)

from pushbullet import PushBullet
import time


'''Importing API keys from SQL database for the corresponding patient's doctor '''


def notification(patient_ID):
    api_query = '''select API_key from notification_api as api inner join patient_data as patient 
                on api.Doctor_ID = patient.Doctor_ID where patient.patientID = %s ;'''
    param = (patient_ID,)
    api_token = pd.read_sql_query(api_query, connection, params = param)
    api_token = str(api_token.loc[0][0]).strip()
    text = 'Patient ID '+ patient_ID + ' has a risk of heart disease, kindly check and take appropriate measures'
    
    pb = PushBullet(api_token)
    push = pb.push_note('Alert', text)


# ### Creating the WebApp

# In[ ]:


from pywebio.input import input
from pywebio.output import put_text, put_table, put_html


# ## Creating system to take values

# In[ ]:



#Reading Patient data from 
patient_data = pd.read_sql_query('select * from patient_data', connection)
patient_data = patient_data.drop('Doctor_ID', 1)



def update_values(value,index, update, patient_data, given_ID, dummy_col):
    for x in dummy_col:
        patient_data.loc[index,x] = 0
    
    patient_data.loc[index,update] = value
    display(patient_data.loc[patient_data['PatientID'] == given_ID])
    updated_values = (patient_data.loc[patient_data['PatientID'] == given_ID])
    updated_values = updated_values.drop('PatientID', 1)
    #patient_data.to_csv('patient_data.csv', index = False)
    done = 1
    param = 1
    
    return updated_values, done



#Function to update the patient records

def update_patient(given_ID):
    count = 0 #Given ID in patient_data
    param = 0
    done = 0 #Values updated
    for x in patient_data['PatientID']:
        if x == given_ID:
            put_text('\nPatient ID Found')
            put_html((patient_data.loc[patient_data['PatientID'] == given_ID]).to_html(border = 0))
            index = patient_data.index[patient_data['PatientID'] == given_ID]
            count = 1
            while param == 0:
                update = input('Please enter the parameter to update - ')
                if update in patient_data.columns:
                    dummy_chest = ['ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA']
                    dummy_ECG = ['RestingECG_LVH','RestingECG_ST']
                    dummy_Angina = ['ExerciseAngina_Y']
                    dummy_slope = ['ST_Slope_Flat', 'ST_Slope_Up']
                    
                   
                    value = input('Enter value to update - ')
                    if update in dummy_chest:
                        updated_values, done = update_values(value,index, update, patient_data, given_ID, dummy_chest)
                        param = 1
                        return updated_values, done
                    elif update in dummy_ECG:
                        updated_values, done = update_values(value,index, update, patient_data, given_ID, dummy_ECG)
                        param = 1
                        return updated_values, done
                    elif update in dummy_Angina:
                        updated_values, done = update_values(value,index, update, patient_data, given_ID, dummy_Angina)
                        param = 1
                        return updated_values, done
                    elif update in dummy_slope:
                        updated_values, done = update_values(value,index, update, patient_data, given_ID, dummy_slope)
                        param = 1
                        return updated_values, done
                    else:
                        patient_data.loc[index,update] = value
                        display(patient_data.loc[patient_data['PatientID'] == given_ID])
                        updated_values = (patient_data.loc[patient_data['PatientID'] == given_ID])
                        updated_values = updated_values.drop('PatientID', 1)
                        done = 1
                        param = 1
                        #patient_data.to_csv('patient_data.csv', index = False)
                        return updated_values, done
                        
                else:
                    put_text('Parameter not found')
            break
    
    if count == 0:
        put_text('Patient ID not found')
    
    return None, done
        


####################################################################
#Main Function

login = False

while login == False:
    os.system('cls')
    data = []
    with open('login.csv') as csvfile:
        reader = csv.reader(csvfile)
        for x in reader:
            data.append(x)
    
    staff_name = input('\nPlease enter your username : ')
    staff_pass = input('Please enter your password: ')

    sname = [x[0] for x in data]
    spass = [x[1] for x in data]

    if staff_name in sname:
        for i in range(0,len(sname)):
            if sname[i] == staff_name and spass[i] == staff_pass:
                put_text('\nYou are logged in')
                login = True
                patient_ID = input('\nEnter patient ID to update the record - ')
                updated_values, done = update_patient(patient_ID)
                countInc = 1
                
                if done == 1:
                    listt = updated_values.values.tolist()
                    B = np.reshape(listt, (-1, 15))
                    listtt = B.tolist()

                    output = RF.predict(listtt)
                    if output[0] == 1:
                        put_text('\n\nALERT! Risk of heart disease.')
                        
                        #Code to send push notifications
                        
                        notification(patient_ID)
                        
                    else:
                        put_text('\nValues updated, no significant chance of heart disease')
                        
                    login = False
                break

