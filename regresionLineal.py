import csv
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

file='datos_moviles.csv'
with open(file) as f:
    lectura=csv.reader(f,delimiter=',')
    encabezados=next(lectura)
    
    data=pd.DataFrame(lectura,columns=encabezados)

#las transformo a int porque estaban en tipo "objeto" y así no se podía calcular la corr    
data['battery_power']=data['battery_power'].astype(int)
data['blue']=data['blue'].astype(int)
data['clock_speed']=data['clock_speed'].astype(int)
data['dual_sim']=data['dual_sim'].astype(int)
data['fc']=data['fc'].astype(int)
data['four_g']=data['four_g'].astype(int)
data['int_memory']=data['int_memory'].astype(int)
data['m_dep']=data['m_dep'].astype(int)
data['mobile_wt']=data['mobile_wt'].astype(int)
data['n_cores']=data['n_cores'].astype(int)
data['pc']=data['pc'].astype(int)
data['px_height']=data['px_height'].astype(int)
data['px_width']=data['px_width'].astype(int)
data['ram']=data['ram'].astype(int)
data['sc_h']=data['sc_h'].astype(int)
data['sc_w']=data['sc_w'].astype(int)
data['talk_time']=data['talk_time'].astype(int)
data['three_g']=data['three_g'].astype(int)
data['touch_screen']=data['touch_screen'].astype(int)
data['wifi']=data['wifi'].astype(int)
data['price_range']=data['price_range'].astype(int)
data['price']=data['price'].astype(int)

#obtengo las 5 correlaciones más fuertes de la variable price_range
#corr_pr_rg=data.corr().iloc[::,20:21]
#print(corr_pr_rg.sort_values(by='price_range',ascending=False).head(6))

#columnas=['battery_power','ram','price']
#sbn.pairplot(data[columnas])
#plt.tight_layout()
#plt.show()

X=np.array(data['ram'])
X=X.reshape(-1, 1)
y=np.array(data['price'])

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=20)
lr=LinearRegression().fit(X_train,y_train)
y_pred=lr.predict(X_test)

print("Coeficiente w1:",lr.coef_)
print("Coeficiente w0:",lr.intercept_)
print("Valor del coeficiente de determinación del conjunto de entrenamiento:",round(r2_score(X_train,y_train),3))
print("Valor del coeficiente de determinación del conjunto de prueba:",round(r2_score(X_test,y_test),3))
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('RAM')
plt.ylabel('Precio')
plt.title('Regresión lineal Precio=f(RAM)')
plt.show()

valor=np.array(3100)
valor=valor.reshape(1,-1)
valor_pred=lr.predict(valor)
print(valor_pred)


residuos=y_test-y_pred
plt.scatter(y_pred, residuos,c='steelblue',marker='o',edgecolor='white',
label='Datos de entrenamiento')
plt.xlabel('Predicción')
plt.ylabel('Residuos')
plt.hlines(y=0,xmin=20,xmax=2000,color='black',lw=2)
plt.xlim([20,2000])
plt.show()