!pip install outlier_utils
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing 
import statsmodels.api as sm 
import numpy as np
from outliers import smirnov_grubbs as grubbs 

# Cargar el archivo CSV
df = pd.read_csv("credito.csv")

# Xminmax para todas las variables
minimos = df.min() 
maximos = df.max()

for name, values in df.items():
    Xminmax = (df[name] - minimos[name]) / (maximos[name] - minimos[name])
    print(Xminmax)

# Alternativa usando sklearn
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df)
df_scaled = pd.DataFrame(x_scaled, columns=df.columns)
print(df_scaled)

# Estandarización normal
df_std = pd.DataFrame()
for name, values in df.items():
    df_std[name] = (df[name] - df[name].mean()) / df[name].std()
    print(df_std[name])

# Alternativa usando sklearn
scaler = preprocessing.StandardScaler()
df_std = scaler.fit_transform(df) 
df_std = pd.DataFrame(df_std, columns=df.columns)
print(df_std)

# Gráfico de caja para 'monto'
plt.boxplot(df['monto'])
plt.title("Diagrama de caja de Monto")
plt.show()

# Diagrama de dispersión entre 'monto' y 'edad'
plt.figure()
plt.scatter(df['monto'], df['edad'])
plt.xlabel('Monto')
plt.ylabel('Edad')
plt.title('Diagrama de dispersión de Monto vs Edad')
plt.show()

# Histograma de 'monto'
plt.figure()
fig, ax = plt.subplots(figsize=(10, 7))
ax.hist(df['monto'], bins=10)
plt.title("Histograma de Monto")
plt.show()

# QQ-plot de 'monto'
plt.figure()
fig = sm.qqplot(df['monto'], line='45')
plt.title("QQ-plot de Monto")
plt.show()

# QQ-plot de datos aleatorios
data = np.random.normal(0,1, 1000)
plt.figure()
fig = sm.qqplot(data, line='45')
plt.title("QQ-plot de datos aleatorios")
plt.show()

# Test de Grubbs para 'monto'
test = grubbs.test(df['monto'], alpha=.05) 
indexes = grubbs.max_test_indices(df['monto'], alpha=.05)
values = grubbs.max_test_outliers(df['monto'], alpha=.05)
print(indexes)
print(values)
print(test)

# Correlaciones
pearson = df.corr(method='pearson')
spearman = df.corr(method='spearman')
kendall = df.corr(method='kendall')
print(pearson)
print(spearman)
print(kendall)
