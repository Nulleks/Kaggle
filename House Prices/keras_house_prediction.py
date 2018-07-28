# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 01:08:46 2018

@author: 0x
"""

import pandas as pd
import seaborn as sns
import numpy as np

sns.set_style('darkgrid')

# Read csv
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
ids = test['Id'].values
len_train = len(train)
gender_submission_data = pd.read_csv('sample_submission.csv')
test = pd.merge(test, gender_submission_data, on='Id')
all_data = pd.concat((train,test), sort=False).reset_index(drop=True)




all_data_null = (all_data.isnull().sum() / len(all_data)) * 100
all_data_null = all_data_null.drop(all_data_null[all_data_null == 0].index).sort_values(ascending=False)


"""
print(train.sample(5))
print(train.isnull().sum())
print(test.isnull().sum())
sns.countplot(x='SalePrice', data=train, hue='Fence')
sns.lineplot(x='GrLivArea', data=train, y='SalePrice')
sns.pointplot(x='GrLivArea', data=train, y='SalePrice')
sns.distplot(train['SalePrice'])
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
"""

def preprocess_data(df):
    df.drop('Id', axis=1, inplace=True)
    
    #Not nan values, nan means None = it doesnt have the quality like pool or fence
    random_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
    for column in random_none:
        df[column].fillna('None', inplace=True)
    garage_none  = ['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType']
    for column in garage_none:
        df[column].fillna('None', inplace=True)
    basement = ['BsmtCond', 'BsmtExposure','BsmtQual', 'BsmtFinType2','BsmtFinType1']
    for column in basement:
        df[column].fillna('None', inplace=True)
    
    # Bucar cosas raras, si un campo dice que no tiene sotano ejem!
    
    df["PoolQC"].fillna("None", inplace=True)
    df["MiscFeature"].fillna("None", inplace=True)
    df["Alley"].fillna("None", inplace=True)
    df["Fence"].fillna("None", inplace=True)
    df["FireplaceQu"].fillna("None", inplace=True)    


    median = df.groupby("Neighborhood")["LotFrontage"].median()
    df['LotFrontage'] = df.apply(lambda row: median[row['Neighborhood']] if np.isnan(row.LotFrontage) else row.LotFrontage, axis=1)

    garage_fill = ['GarageYrBlt','GarageArea','GarageCars']
    for column in garage_fill:
        df[column].fillna(0, inplace=True)
    # GarageYrBlt año en el que fue construido el garage, igual la mediana tambien del barrio, numerical
    # GarageArea tamaño del garage
    # GarageCars tamaño del garage en coches, si garageArea es 0 este es 0 este es delicado
    
    # MasVnrType tipo de chapa o enchapado, mas comun del barrio?, categorical
    # MasVnrArea area del chapado mediana? , numerica
    df['MasVnrType'].fillna('None', inplace=True)
    df['MasVnrArea'].fillna(0, inplace=True)
    
    
    # MSZoning clasificacion de la zona, mas comun, categorical
    #sns.countplot(x='MSZoning', data=df)
    df['MSZoning'].fillna(df['MSZoning'].mode()[0], inplace=True)
    
    basement_fill = ['BsmtFullBath','BsmtHalfBath','TotalBsmtSF','BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF']
    for column in basement_fill:
        df[column].fillna(0, inplace=True)
    # BsmtFullBath nº de baños del basement? mas comun?, numerical
    # BsmtHalfBath, numerical
    # TotalBsmtSF, tamaño del basement, si no tiene esto es 0
    # BsmtFinSF2 metros finalizados, numerical
    # BsmtUnfSF metros sin finalizar numerical    
    
    
    # Functional deduccion segun estado,  mas comun, categorical
    df['Functional'].fillna(df['Functional'].mode()[0], inplace=True)
    # Utilities mas comun del barrio, categorical / drop column or AllPub, non relevant column
    df.drop('Utilities', axis=1, inplace=True)

    # Electrical mas comun, categorical
    df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
    
    # Exterior1st, mas comun, categorical
    # Exterior2nd igual que el 1º
    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])

    # KitchenQual calidad de la cocina, categorical 
    df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
    # SaleType tipo de venta, mas comun del barrio
    df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
   
    
    return df
    

 """   
null_data = all_data[all_data.isnull().any(axis=1)]   
sns.countplot(x='MSZoning', data= all_data)
sns.countplot(x='Functional', data= all_data)
sns.countplot(x='Electrical', data= all_data)
sns.countplot(x='Exterior1st', data= all_data)
a = train.groupby(["Neighborhood",'Exterior1st']).size()
a = train.groupby(["LotFrontage"]).size()
print(train.isnull().sum())
print(test.isnull().sum())
train_null = (train.isnull().sum() / len(train)) * 100
train_null = train_null.drop(train_null[train_null == 0].index).sort_values(ascending=False)
"""


all_data = preprocess_data(all_data)










def convert_to_categorical(df):    
    #categorical_onehot = ['MSSubClass','MSZoning', 'Street','Alley','LotShape', 'LandContour', 'LandSlope','Neighborhood','Condition1','Condition2']
    
    df['MSSubClass'] = df['MSSubClass'].apply(str)
    df = pd.get_dummies(df,drop_first=True)    

    #number_to_categorical = ['LotFrontage','LotArea']
    return df


all_data = convert_to_categorical(all_data)

x_train, x_test = all_data.iloc[:len_train, :], all_data.iloc[len_train:, :]
y_train = x_train['SalePrice'].values
x_train = x_train.drop('SalePrice', axis=1, inplace=False)

y_test = x_test['SalePrice'].values
x_test = x_test.drop('SalePrice', axis=1, inplace=False)





from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(all_data.drop('SalePrice', axis=1))
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)





# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)
classifier.score(x_train,y_train)
acc_random_forest = round(classifier.score(x_train, y_train) * 100, 2)
# Predicting the Test set results
y_pred = classifier.predict(x_test)



ids = ids.reshape(-1,1)
ids = np.squeeze(ids)
y_pred = np.squeeze(y_pred)
output = pd.DataFrame({ 'Id' : ids, 'SalePrice': y_pred })

output.to_csv('house_prediction_Forest.csv', index = False)








# Part 2 - Now let's make the ANN!
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics


from sklearn.preprocessing import StandardScaler
scPrice = StandardScaler()
scPrice.fit(all_data['SalePrice'].values.reshape(-1,1))
y_train = scPrice.transform(y_train.reshape(-1,1))
y_test = scPrice.transform(y_test)

def build_classifier():
    # Initialising the ANN
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    # Numer or nodes are art, but here we are taking,number of ind variables+ dependent variables --> 11+1 /2
    classifier.add(Dense(units = 124, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    # Adding the second hidden layer
    classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 124, kernel_initializer = 'uniform', activation = 'relu'))
        
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform'))
    # Compiling the ANN
    #adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = [metrics.mae])  # if we have more than two categories, categorical_crossentropy
    return classifier



classifier = build_classifier()
classifier.fit(x_train, y_train, batch_size = 32, epochs =50, verbose=1, validation_split=0.2) # 0.8058


y_pred2 = classifier.predict(x_test)
y_pred2 = scPrice.inverse_transform(y_pred2)
y_pred2 = np.squeeze(y_pred2)

output = pd.DataFrame({ 'Id' : ids, 'SalePrice': y_pred2 })

output.to_csv('house_prediction_NN.csv', index = False)








