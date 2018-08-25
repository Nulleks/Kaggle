# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 00:19:22 2018

@author: 0x
"""

import numpy as np # linear algebra
import pandas as pd # data processing
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




from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(testing_data)


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)


##### Pytorch NN ######

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(18, 10)
        self.fc2 = nn.Linear(10, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        
        return x
    
    
net = Net()



batch_size = 50
num_epochs = 50
learning_rate = 0.01
batch_no = len(x_train) // batch_size


#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#one_hot_encoder = OneHotEncoder(categorical_features = [0])
#y_train = one_hot_encoder.fit_transform(y_train).toarray()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


from sklearn.utils import shuffle
from torch.autograd import Variable

for epoch in range(num_epochs):
    if epoch % 5 == 0:
        print('Epoch {}'.format(epoch+1))
    x_train, y_train = shuffle(x_train, y_train)
    # Mini batch learning
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        x_var = Variable(torch.FloatTensor(x_train[start:end]))
        y_var = Variable(torch.FloatTensor(y_train[start:end]))
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        ypred_var = net(x_var)
        loss =criterion(ypred_var, y_var)
        loss.backward()
        optimizer.step()

       
# Evaluate the model
test_var = Variable(torch.FloatTensor(x_val), requires_grad=True)
with torch.no_grad():
    result = net(test_var)

result = np.where(result > 0.5, 1, 0)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, result)


