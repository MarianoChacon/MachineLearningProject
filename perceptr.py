import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


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

ppn=Perceptron(max_iter=40,eta0=1,random_state=1)
ppn.fit(X_train_std,y_train)
predic=ppn.predict(X_test_std)

print(f"The accuracy is {accuracy_score(y_true = y_test, y_pred = predic, normalize = True)}")