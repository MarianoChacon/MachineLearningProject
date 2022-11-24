import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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

svm_lin=LinearSVC()
svm_lin.fit(X_train,y_train)

pred=svm_lin.predict(X_test)
print(f"The accuracy of the model is {accuracy_score(y_true=y_test,y_pred=pred,normalize=True)}")
#---------variance
print(df.var())
#---------variables standarization
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.fit_transform(X_test)


svm_lin=LinearSVC()
svm_lin.fit(X_train_std,y_train)

pred=svm_lin.predict(X_test_std)
print(f"The accuracy of the model is {accuracy_score(y_true=y_test,y_pred=pred,normalize=True)}")

#--------------Changin C value

c_values=[0.01,0.05,0.1]
accuracies=[]
for k in c_values:
    svm_lin=LinearSVC(C=k)
    svm_lin.fit(X_train_std,y_train)

    pred=svm_lin.predict(X_test_std)
    print(f"The accuracy of the model with C={k} is {accuracy_score(y_true=y_test,y_pred=pred,normalize=True)}")
    accuracies.append(accuracy_score(y_true=y_test,y_pred=pred,normalize=True))


plt.plot(c_values,accuracies,marker="o",color="blue")
plt.xticks(c_values)
plt.xlabel("C values")
plt.ylabel=("accuracy")
plt.axvline(x=0.01,linestyle='--',color="grey")
plt.title("Accuracy changing C parameter")
plt.show()
