import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# nltk.download()
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')

from sklearn.linear_model import LinearRegression

# Opção para visualizar todas as colunas
pd.set_option('display.max_columns', None)

# Leitura do csv
df = pd.read_csv('realtor-data.csv', low_memory=False)

# Analise do dataset
print("Primeiros registos: ")
print(df.head(5))
print("\n")

print("Tamanho (rows,colunas): ")
print(df.shape)
print("\n")

print("Info e Tipo de dados: ")
print(df.info())
print("\n")

# Análise por coluna
print("Análise por coluna:")

for col in df.columns:
    print("Coluna: " + col)
    print("Nº de valores distintos em " + col + ": " + str(len(df[col].value_counts())))
    print("Valores presentes em " + col + ": " + str(df[col].unique()))
    print("Nº de valores nulos em " + col + ": " + str(sum(df[col].isna())))
    print("\n")

# Eliminar colunas
df = df.drop(columns=["sold_date", "street", "full_address"])
print("As colunas sold_date, street e full_address foram eliminadas!")
print("\n")

# Eliminar valores nulos
df = df[df['price'].notna()]
print("Os valores nulos de price foram eliminados!")
print("\n")

df = df[df['city'].notna()]
print("Os valores nulos de city foram eliminados!")
print("\n")

df = df[df['zip_code'].notna()]
print("Os valores nulos de city foram eliminados!")
print("\n")

# Alteração de Data Types
# Valor com erro na linha 1399 e 732
df = df.drop(1399)
df = df.drop(732)

# Para float64 (numeric)
df["price"] = pd.to_numeric(df["price"])
df["bed"] = pd.to_numeric(df["bed"])
df["house_size"] = pd.to_numeric(df["house_size"], errors='coerce')

#Elimina valores duplicados
df.drop_duplicates(ignore_index=True, inplace=True)
print("Valores duplicados foram eliminados!")
print("\n")

print("(rows, colunas):")
print(df.shape)
print("\n")

# Troca de valores com erros na coluna status
df['status'] = df['status'].replace(['for_salee'], 'for_sale')
df['status'] = df['status'].replace(['for_ssale'], 'for_sale')
df['status'] = df['status'].replace(['for_sale'], '1')
df['status'] = df['status'].replace(['ready_to_build'], '0')

# Para float64 (numeric)
df["status"] = pd.to_numeric(df["status"])

# Remover terrenos para construir
df = df[df['status'] > 0]

# Data types
print("Os tipos de dados das colunas foram alterados e os 'ready_to_build' foram removidos!")
print("\n")

tipos = df.dtypes
print("Data Type de cada coluna:")
print(tipos)
print("\n")

# Analise box plot
print("Análise antes de remover outliers: ")
print(df.describe())
print("\n")

df.plot.box(column=['price'], grid='True')
df.plot.box(column=['bed', 'bath'], grid='True')
df.plot.box(column=['acre_lot'], grid='True')
df.plot.box(column=['house_size'], grid='True')
plt.show()

# Remover Outliers

# Valores negativos e outliers
df['bed'] = df['bed'].replace([-2], 2)
df['acre_lot'] = df['acre_lot'].replace([100000], np.nan)
df['acre_lot'] = df['acre_lot'].replace([99999], np.nan)
df['acre_lot'] = df['acre_lot'].replace([96120], np.nan)
df['price'] = df['price'].replace([875000000], np.nan)
df['price'] = df['price'].replace([0], np.nan)
df = df[df.price > 1000]
df = df[df.price != 169000000]
df = df[df.bath != 198]
df = df[df.bath != 123]
df = df[df.bed != 86]
df['house_size'] = df['house_size'].replace([-999], np.nan)
df = df[df.house_size != 1450112]
df = df[df.house_size != 400149]

print("Análise genérica após remover outliers: ")
print(df.describe())
print("\n")

#Box plot após remover outliers
df.plot.box(column=['price'], grid='True')
plt.ylim(1000, 1900000)
df.plot.box(column=['bed', 'bath'], grid='True')
plt.ylim(0, 10)
df.plot.box(column=['acre_lot'], grid='True')
plt.ylim(0, 3)
df.plot.box(column=['house_size'], grid='True')
plt.ylim(10, 5000)
plt.show()

# Analise após limpeza
print("Neste momento, o dadaframe possui (rows,colunas): ")
print(df.shape)
print("\n")

# for col in df.columns:
#    print("Coluna: " + col)
#    print("Nº de valores distintos em " + col + ": " + str(len(df[col].value_counts())))
#    print("Valores presentes em " + col + ": " + str(df[col].unique()))
#    print("Nº de valores nulos em " + col + ": " + str(sum(df[col].isna())))
#    print("\n")

# Uso de Regressão linear
print('Analise de valores nulos no dataframe')
print(df.isnull().sum())
print("\n")

# Criar Dataframe com valores nulos em "house_size"
test = df[df['house_size'].isnull()]
test = test[test['bed'].notna()]
test = test[test['bath'].notna()]
test = test[test['acre_lot'].notna()]

print('Foi criado um subset sem valores nulos!')
print("\n")

# Drop valores nulos no dataframe
df1 = df.drop(columns=['status', 'city', 'state', 'zip_code'])
df1.dropna(inplace=True)
print('Subset sem valores nulos (e sem as colunas tipo string e zip_code)')
print(df1)
print("\n")

print('Analise de valores bulos do subset')
print(df1.isnull().sum())
print("\n")

# Aplicação do modelo no subset onde os valores de house_size são nulos
x_train = df1.drop('house_size', axis=1)
y_train = df1['house_size']

print('A coluna "house_size" foi considerada como váriável dependente para criar modelo de LR!')

# Construi o modelo de regressão linear

lr = LinearRegression()

lr.fit(x_train, y_train)

print('Modelo de Regressão Linear criado!')
print("\n")

# Criar subset onde os dados da coluna house_size são nulos
df_house_null = test[['price', 'bed', 'bath', 'acre_lot']]
df_house_null.dropna(inplace=True)


# Aplicar o modelo no x_test para fazer previsões
y_pred = lr.predict(df_house_null)

print('Modelo de Regressão Linear aplicado!')
print('Valores gerados:')
print(y_pred)
print("\n")

# Troca de valores nulos pelos valores previstos
test['y_pred'] = y_pred
#test.to_csv('y_previsto.csv')

df = df.combine_first(test)

#Arredonda valores de y_pred
df.y_pred = df.y_pred.round()

#Acresenta valores de y_pred a house_size
df['housesize'] = df.apply(lambda row: row.house_size if pd.notnull(row.house_size)
                       else row.y_pred, axis = 1)

df = df.drop(['y_pred', 'house_size'], axis=1)

print('valores previstos pelo modelo foram aplicados no DataFrame!')
print("\n")
print('Formato final do Dataframe:')
print(df)

#df.to_csv('DataFrame_AfterLR.csv')
#print('DataFrame gravado em csv!')