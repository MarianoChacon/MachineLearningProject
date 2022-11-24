import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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
logit_reg=LogisticRegression()
logit_reg.fit(X_train,y_train)

print(f"Intercept: {logit_reg.intercept_}")
print(f"Coeficients: Age, {logit_reg.coef_[0][0]}; Estimated Salary, {logit_reg.coef_[0][1]};Gender, {logit_reg.coef_[0][2]}")

pred=logit_reg.predict(X_test)
print(f"The accuracy of the model is {accuracy_score(y_true=y_test,y_pred=pred,normalize=True)}")
