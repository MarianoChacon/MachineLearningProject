import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from pydotplus import graph_from_dot_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2)

accuracies=[]
for i in range(1,10):
    tree_clf1=DecisionTreeClassifier(max_depth=i)
    tree_clf1.fit(X_train,y_train)
    predict1=tree_clf1.predict(X_test)
    print(f"Accuracy of the model depth={i} {accuracy_score(y_true=y_test,y_pred=predict1,normalize=True)}")
    accuracies.append(accuracy_score(y_true=y_test,y_pred=predict1,normalize=True))


plt.plot(range(1,10),accuracies)
plt.title("Accuracies comparison for distinct depth values")
plt.xlabel("Depth")
plt.ylabel("Accuracies")
plt.show()

tree_clf=DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X_train,y_train)
predict=tree_clf.predict(X_test)

tree_data=export_graphviz(tree_clf,feature_names=['Age','EstimatedSalary','gen_bin'],out_file=None,rounded=True)
graphic=graph_from_dot_data(tree_data)
graphic.write_png("treeDepth2NOfill.png")



pred_example=tree_clf.predict(np.array([35,35000,0]).reshape(1,-1))
if pred_example==0:
    print(f"The model predicts that a woman with salary of 35000 and 35 years old is not going to purchase")
elif pred_example==1:
    print(f"The model predicts that a woman with salary of 35000 and 35 years old is going to purchase")