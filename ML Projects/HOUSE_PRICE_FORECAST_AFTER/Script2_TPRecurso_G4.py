import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import joblib

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso
import shap
import warnings
warnings.filterwarnings('ignore')

# Opção para visualizar todas as colunas
pd.set_option('display.max_columns', None)

# Leitura do csv
df = pd.read_csv('DataFrame_AfterLR.csv')

# Remoção de colunas
df = df.drop(['Unnamed: 0', 'status', 'zip_code', 'city'], axis=1)
print('As colunas "status", "zip_code" e "city" foram removidas!')
print("\n")

# Matriz de correlação
print('Matriz de correlação:')
print(df.corr())
print("\n")

# Valores antes de conversão
print('Data Frame antes da conversão de USD para €, acres para hectares e sqft para m^2:')
print(df.head())
print("\n")

# Converte de USD para EUR (valor de 13/02/2023)
df['price'] = df['price'] * 0.93

# Converte acres em hectares
df['acre_lot'] = df['acre_lot'] * 0.404

# Converte square feet em metros quadrados
df['housesize'] = df['housesize'] * 0.09290304

# Arredonda valores de y_pred
df.housesize = df.housesize.round()
print('Os valores de house_size foram arredondados!')
print("\n")

# Troca do nome de colunas
df.rename(columns={"acre_lot": "hectares"}, inplace=True)
df.rename(columns={"housesize": "house_size"}, inplace=True)

# Dataframe após a conversão
print('Data Frame após a conversão')
print(df.head())
print("\n")

print('Valores nulos no dataframe')
print(df.isna().sum())
print("\n")

# Analise
print("Neste momento, o dadaframe possui (rows,colunas): ")
print(df.shape)
print("\n")

# Remove valores nulos de hectares
df = df[df['hectares'].notna()]

# Analise após limpeza
print("Valores valores nulos de hectares eliminados! ")
print("\n")

# Analise
print("Neste momento, o dadaframe possui (rows,colunas): ")
print(df.shape)
print("\n")

print('Valores nulos no dataframe')
print(df.isna().sum())
print("\n")

# Matriz de correlação visual
plt.figure(dpi=100)
plt.title('Correlation Matrix')
sns.heatmap(df.corr(), annot=True, lw=1, linecolor='white', cmap='viridis')
plt.xticks(rotation=60)
plt.yticks(rotation=60)
plt.title('Matriz de Correlação')
plt.show()

# Quantidade de estados presentes
plt.figure(figsize=(20, 7))
sns.countplot(x=df["state"])
plt.xticks(rotation=90)
plt.title('Número de casas por estado')
plt.ylabel('Nº de casas')
plt.show()

# Remoção de estados com poucos registos
df = df[df.state != 'Tennessee']
df = df[df.state != 'Wyoming']
df = df[df.state != 'Virginia']
df = df[df.state != 'Georgia']
df = df[df.state != 'West Virginia']

print('Estados Tennessee, Wyoming, Virginia, Georgia e West Virginia foram removidos!')
print("\n")

# Preço médio por estado
df.groupby("state")["price"].mean().round(2).plot(kind="bar")
plt.title('Preço médio por estado')
plt.ylabel('Valor médio (em €)')
plt.show()

# Preparação do dataset para aplicação de modelos
df1 = df.dropna()
print('Valores nulos no dataframe eliminados para aprendizagem!')
print("\n")

print('Valores nulos no dataframe para treino e teste:')
print(df1.isna().sum())
print("\n")

print('(rows,colunas):')
print(df1.shape)
print("\n")

print('Valores na coluna state:')
print(df1['state'])
print("\n")

# Coloca labels (1,2,3,..., n) na coluna state
LB = LabelEncoder()
df1["state"] = LB.fit_transform(df1["state"])

print('Coluna state convertida em labels (integer)!')
print("\n")

print('Registos da coluna state:')
print(df1['state'])
print('Coluna state convertida em labels!')
print(df1)
print("\n")

df1.to_csv('DF_treinos.csv')

tipos1 = df1.dtypes
print("Data Type de cada coluna:")
print(tipos1)
print("\n")

# Separação dos dados em variaveis independetes (X) e dependentes (Y)
X = df1[['hectares', 'bath', 'bed', 'house_size', 'state']]
y = df1['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

print("Dataframe separado em dados de treino e teste!")
print("\n")

print("Valores em cada dataframe:")
print("X Train : ", X_train.shape)
print("X Test  : ", X_test.shape)
print("Y Train : ", y_train.shape)
print("Y Test  : ", y_test.shape)
print("\n")

# Fit the KNN model
knn = KNeighborsRegressor(27)
knn.fit(X_train, y_train)

print('Modelo KNN gerado!')
print("\n")

# Predict the missing values
y_pred = knn.predict(X_test)

print('Modelo KNN testado!')
print("\n")

# Métricas de performance
print('Score do modelo KNN:')
print(knn.score(X, y))
print("\n")

print('Valor do Erro Quadrático Médio (MSE)no modelo KNN:')
print(mean_squared_error(y_test, y_pred))
print("\n")

# Modelo XGB
XGB = XGBRegressor()

# Fit modelo XGB
XGB.fit(X_train, y_train)

print('Modelo XGB gerado!')
print("\n")

# Compara valores previsto com valores de teste
y_pred = XGB.predict(X_test)

print('Modelo XGB testado!')
print("\n")

# Guarda o modelo treinado em ficheiro pkl
joblib.dump(XGB, 'modeloXGB.pkl')
print('Modelo XGB guardado em ficheiro .pkl!')
print("\n")

# Métricas de performance
print('Score para modelo XGB:')
print(XGB.score(X, y))
print("\n")

print('Valor do Erro Quadrático Médio (MSE) no modelo XGB:')
print(mean_squared_error(y_test, y_pred))
print("\n")

# Testagem do modelo
# 'hectares','bath','bed','house_size','state'
inputs = pd.DataFrame({'hectares': 2, 'bath': 7, 'bed': 5, 'house_size': 3000, 'state': 0}, index=[0])

result = XGB.predict(inputs)
print('Valor previsto para um casa de 3000 metros quadrados, com 2 hectares, 7 wcs e  5 camas no estado do Connecticut:')
print(result)

# Modelo K Means
kmeans = KMeans(n_clusters=10, random_state=120)
kmeans.fit(df1)

print('Modelo K-Means gerado, com 10 clusters!')
print("\n")

cluster_map = pd.DataFrame(df1)
cluster_map['cluster'] = kmeans.labels_

print('Dataframe com a coluna cluster:')
print(cluster_map)

cluster_count = cluster_map['cluster'].value_counts()

# plotting the pie chart
plt.pie(cluster_count, labels=cluster_count.index,
        autopct='%1.1f%%', shadow=True)
plt.title('Analise de dimensões com K-Means')
plt.show()


# ==========================================
# 🚀 NEW SECTION: MODEL OPTIMIZATION + EXPLAINABILITY
# ==========================================

print("\n===============================")
print("Iniciando otimização automática de modelos e interpretabilidade...")
print("===============================\n")

# --- Automated Hyperparameter Tuning for KNN ---
param_grid_knn = {
    'n_neighbors': [3, 5, 9, 15, 21, 27, 35],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}
grid_knn = GridSearchCV(KNeighborsRegressor(), param_grid_knn, cv=5,
                        scoring='neg_mean_squared_error', n_jobs=-1)
grid_knn.fit(X_train, y_train)
print("Melhores parâmetros para KNN:", grid_knn.best_params_)
print("Melhor Score (MSE):", -grid_knn.best_score_)
best_knn = grid_knn.best_estimator_

# --- Automated Hyperparameter Tuning for XGBoost ---
param_grid_xgb = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.9, 1.0],
    'reg_lambda': [1, 5, 10]
}
grid_xgb = GridSearchCV(XGBRegressor(random_state=27), param_grid_xgb, cv=5,
                        scoring='neg_mean_squared_error', n_jobs=-1)
grid_xgb.fit(X_train, y_train)
print("Melhores parâmetros para XGBoost:", grid_xgb.best_params_)
print("Melhor Score (MSE):", -grid_xgb.best_score_)
best_xgb = grid_xgb.best_estimator_

print("\nModelos otimizados gerados com sucesso!\n")

# --- Regularization Techniques ---
print("Aplicando regularização (Ridge e Lasso)...\n")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
lasso = Lasso(alpha=0.001)
lasso.fit(X_train, y_train)

print("Ridge Score:", ridge.score(X_test, y_test))
print("Lasso Score:", lasso.score(X_test, y_test))
print("MSE Ridge:", mean_squared_error(y_test, ridge.predict(X_test)))
print("MSE Lasso:", mean_squared_error(y_test, lasso.predict(X_test)))
print("\nRegularização concluída.\n")

# --- Model Interpretability (SHAP) ---
print("Gerando explicações com SHAP...\n")
explainer = shap.Explainer(best_xgb)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("Importância global das features (XGBoost)")
plt.show()

sample_idx = 0
shap.force_plot(explainer.expected_value, shap_values[sample_idx, :],
                X_test.iloc[sample_idx, :], matplotlib=True)
plt.title("Explicação individual da previsão (sample 0)")
plt.show()

print("\nOtimização, regularização e interpretabilidade concluídas com sucesso!\n")
