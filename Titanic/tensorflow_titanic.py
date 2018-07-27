# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 02:28:07 2018

@author: 0x
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_model as tf_model

sns.set_style('whitegrid')



## Creating panda dataframes from train and test CSV files
print("Loading Training and Testing Data =====>")
training_data = pd.read_csv('data/train.csv')
testing_data = pd.read_csv('data/test.csv')
gender_submission_data = pd.read_csv('data/gender_submission.csv')
testing_data = pd.merge(testing_data, gender_submission_data, on='PassengerId')
print("<===== Training and Testing Data Loading finished")




print(testing_data.sample(5))
print (training_data.isnull().sum())
print(training_data.describe())
print (testing_data.isnull().sum())
print(testing_data.describe())


sns.countplot(x='Survived',data=training_data,hue='Sex')
sns.countplot(x='Pclass',data=training_data,hue='Survived')
sns.factorplot(x='Survived',data=training_data,hue='Sex',kind='count',col='Pclass')
sns.distplot(training_data['Age'].dropna(), bins=30, kde=False)
sns.countplot(x='Embarked',data=training_data,hue='Survived')




def name(x):
    titles = ['Mr', 'Miss', 'Mrs','Master']
    if x not in titles:
        return 'other'
    else:
        return x   
    
    
def preprocess_data(df):
    delete_columns = ["PassengerId", "Ticket", "Cabin"]
    df['Name']=df['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
    df['Name']=df['Name'].apply(name)
    df.drop(delete_columns, axis=1, inplace = True)
    
    df['Age'].fillna(df["Age"].mean(), inplace = True)
    df['Embarked'].fillna('S', inplace=True)
    df['Fare'].fillna(df["Fare"].mean(), inplace = True)
    
    df = pd.get_dummies(df, columns=['Pclass'], drop_first=True)
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    df = pd.get_dummies(df, columns=['Name'], drop_first=True)
    
    df['CategoricalAge'] = pd.cut(df['Age'], 5)
    df['CategoricalFare'] = pd.cut(df['Fare'], 4)
    df = pd.get_dummies(df, columns=['CategoricalAge'], drop_first=True)
    df = pd.get_dummies(df, columns=['CategoricalFare'], drop_first=True)
    df.drop("Age", axis=1, inplace=True)
    df.drop("Fare", axis=1, inplace=True)

    
    df['isAlone'] = df.apply(lambda x: 0 if x.SibSp + x.Parch >=1 else 1, axis=1)
    
    df['familySize'] = df.apply(lambda x: x.SibSp + x.Parch + 1, axis=1)
    df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

    
    return df
    




training_data = preprocess_data(training_data)
testing_data = preprocess_data(testing_data)



x_train = training_data.iloc[:, 1:].values
y_train = training_data.iloc[:, :1].values

x_test = testing_data.iloc[:, 1:].values
y_test = testing_data['Survived'].values.reshape(-1,1)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



_, _, parameters, results = tf_model.model(x_train, y_train, x_test, y_test)


results = np.where(results > 0.5, 1, 0)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, results)