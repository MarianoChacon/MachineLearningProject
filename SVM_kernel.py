import pandas as pd
import numpy as np
from sklearn.svm import SVC
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

#--------------Changin kernel

kernel_values=['linear','rbf','poly','sigmoid']
accuracies=[]
for k in kernel_values:
    svm_lin=SVC(kernel=k,C=0.01)
    svm_lin.fit(X_train_std,y_train)

    pred=svm_lin.predict(X_test_std)
    print(f"The accuracy of the model with kernel {k} is {accuracy_score(y_true=y_test,y_pred=pred,normalize=True)}")
    accuracies.append(accuracy_score(y_true=y_test,y_pred=pred,normalize=True))


plt.bar(kernel_values,accuracies,color="blue",tick_label=kernel_values)
plt.xticks(kernel_values)
plt.xlabel("kernel")
plt.ylabel=("accuracy")
plt.title("Accuracy changing kernel")
plt.show()

#----data frame with different Gamma value for each kernel type
ker=[]
gam=[0.1,0.2,0.5,1,5,10,20]
gam_i=[]
accuracies1=[]
for k in kernel_values:
    for i in gam:
        svm_lin1=SVC(kernel=k,C=0.01,gamma=i)
        svm_lin1.fit(X_train_std,y_train)

        pred=svm_lin1.predict(X_test_std)
        print(f"Accuracy kernel {k}, gamma {i} {accuracy_score(y_true=y_test,y_pred=pred,normalize=True)}")
        ker.append(k)
        accuracies1.append(accuracy_score(y_true=y_test,y_pred=pred,normalize=True))
        gam_i.append(i)

col={'kernel':ker,'accuracy':accuracies1,'Gamma':gam_i}
table=pd.DataFrame(data=col)
print(table)

sns.lineplot(data=table, x="Gamma", y="accuracy", hue="kernel")
plt.title("Comparison of kernels for distinct Gamma values")
plt.yticks(table['accuracy'])
plt.show()