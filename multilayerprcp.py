import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as ts
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


lectura=pd.read_csv('Social_Network_ads.csv',delimiter=',')
gender=lectura['Gender']
gen_bin=[]
for i in gender:
    if i == 'Male':
        gen_bin.append(1)
    else:
        gen_bin.append(0)

df_gen=pd.DataFrame({'gen_bin':gen_bin})
df=lectura.join(df_gen)
X = pd.DataFrame(np.c_[df['Age'], df['EstimatedSalary'],df['gen_bin']], columns = ['Age','EstimatedSalary','gen_bin'])

y=df['Purchased']


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=20)

sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.fit_transform(X_test)

model=Sequential()
model.add(Dense(units=50,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics='accuracy')
model.fit(X_train_std,y_train,epochs=200,batch_size=32)

predic=model.predict(X_test_std)

predic=[0 if val <0.5 else 1 for val in predic]

print(f"The accuracy is {accuracy_score(y_true = y_test, y_pred = predic, normalize = True)}")