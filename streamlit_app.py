# importar bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# carregar dados em um DataFrame
df = pd.read_csv('dados_fotovoltaico.csv')

# limpar dados
df.dropna(inplace=True)
df.fillna(0, inplace=True)

# explorar dados
print(df.describe())

# dividir os dados em conjuntos de treinamento e teste
X = df[['Irradiacao', 'Temperatura']]
y = df['energia_gerada']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# criar e treinar o modelo
reg = LinearRegression()
reg.fit(X_train, y_train)

# fazer previsões
y_pred = reg.predict(X_test)

# avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R2 Score:', r2)

#criar grafico de linhas
df = pd.read_csv("dados_fotovoltaico.csv")
plt.plot(df['Data_e_hora'], df['Irradiacao'])
plt.xlabel('Data e Hora')
plt.ylabel('Irradiação (W/m²)')
plt.show()

#criar grafico de barras: para comparar a irradiação ou a energia gerada entre diferentes dias ou entre diferentes horas do dia, você pode usar um gráfico de barras.
df = pd.read_csv("dados_fotovoltaico.csv")
df['Dia_da_semana'] = df['Data_e_hora'].dt.weekday_name
sns.barplot(x='Dia_da_semana', y='Irradiacao', data=df)
plt.xlabel('Dia da semana')
plt.ylabel('Irradiação (W/m²)')
plt.show()

#criar grafico de dispersão: para verificar se há alguma relação entre duas variáveis, como a irradiação e a temperatura, você pode usar um gráfico de dispersão.
df = pd.read_csv("dados_fotovoltaico.csv")
plt.scatter(df['Irradiacao'], df['Temperatura'])
plt.xlabel('Irradiação (W/m²)')
plt.ylabel('Temperatura (°C)')
plt.show()
