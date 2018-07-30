# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 20:30:59 2018

@author: 0x
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')



## Creating panda dataframes from train and test CSV files
print("Loading Training and Testing Data =====>")
training_data = pd.read_csv('data/train.csv')
testing_data = pd.read_csv('data/test.csv')
ids = testing_data.iloc[ : , 0].values

gender_submission_data = pd.read_csv('data/gender_submission.csv')
#testing_data = pd.merge(testing_data, gender_submission_data, on='PassengerId')
print("<===== Training and Testing Data Loading finished")




print(testing_data.sample(5))
print (training_data.isnull().sum())
print(training_data.describe())
print (testing_data.isnull().sum())
print(testing_data.describe())

#correlation matrix
corrmat = training_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


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

#x_test = testing_data.iloc[:, 1:].values
#y_test = testing_data['Survived'].values.reshape(-1,1)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(testing_data)

#unique, counts = np.unique(test_predict_result, return_counts=True)
#dict(zip(unique, counts))








############################ Random Forest ###################################### 
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
classifier.fit(x_train, y_train.ravel())

classifier = RandomForestClassifier()
classifier.fit(x_train, y_train.ravel())

classifier.score(x_train,y_train)
acc_random_forest = round(classifier.score(x_train, y_train) * 100, 2)
# Predicting the Test set results
y_pred = classifier.predict(x_test)

ids = ids.reshape(-1,1)
ids = np.squeeze(ids)
y_pred = np.squeeze(y_pred)
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': y_pred })

output.to_csv('titanic-predictionsForest.csv', index = False)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



############################ Keras NN ###################################### 
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


def build_classifier():
    # Initialising the ANN
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    # Numer or nodes are art, but here we are taking,number of ind variables+ dependent variables --> 11+1 /2
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    # Adding the second hidden layer
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # Compiling the ANN
    #adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])  # if we have more than two categories, categorical_crossentropy
    return classifier




classifier = build_classifier()
classifier.fit(x_train, y_train, batch_size = 32, epochs =100, verbose=1, validation_split=0.2) # 0.8058
# Predicting the Test set results
y_pred = classifier.predict(x_test)
# It return probabilities so if y_pred larger than 0.5 it return true if not it return false
y_pred = np.where(y_pred > 0.5, 1, 0)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred)


y_pred = np.squeeze(y_pred)
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': y_pred })

output.to_csv('titanic-predictionsNN.csv', index = False)






############################ Keras Grid Search ###################################### 
def build_classifier_grid(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 18))
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = "binary_crossentropy",metrics = ['accuracy'])
    return classifier


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
classifier = KerasClassifier(build_fn=build_classifier_grid)
parameters = {"batch_size" : [24,32], "epochs" : [50],"optimizer" : ['adam','rmsprop']} 
grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,scoring="accuracy",cv=10)

grid_search = grid_search.fit(x_train,y_train)

#Best accuracy and parameters
best_parameters =  grid_search.best_params_
best_accuracy = grid_search.best_score_


############################ Keras K fold ###################################### 
# K-Fold Cross Validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
classifier = KerasClassifier(build_fn=build_classifier,batch_size=32, epochs =50)
accuracies = cross_val_score(estimator=classifier,X = x_train, y = y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()


############################ Xgboost ###################################### 

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
# It return probabilities so if y_pred larger than 0.5 it return true if not it return false
y_pred = np.where(y_pred > 0.5, 1, 0)
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': y_pred })

output.to_csv('titanic-xgboost.csv', index = False)

