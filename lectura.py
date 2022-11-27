import pandas as pd
import mglearn

lectura=pd.read_csv('Social_Network_ads.csv',delimiter=',')

#print(f"rows quantity: {len(lectura.axes[1])}\n columns quiantity: {len(lectura.axes[0])}")
#print(lectura.head(5))

gender=lectura['Gender']
gen_bin=[]
for i in gender:
    if i == 'Male':
        gen_bin.append(1)
    else:
        gen_bin.append(0)