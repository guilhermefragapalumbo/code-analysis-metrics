import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
import shap
import joblib  # for parallel processing if needed

pd.set_option('display.max_columns', None)

# --- Load CSV efficiently ---
df = pd.read_csv('realtor-data.csv', low_memory=False)

# --- Initial Cleaning (preserve original steps) ---
df = df.drop(columns=["sold_date", "street", "full_address"])
df = df[df['price'].notna() & df['city'].notna() & df['zip_code'].notna()]
df = df.drop([1399, 732])

df["price"] = pd.to_numeric(df["price"])
df["bed"] = pd.to_numeric(df["bed"])
df["house_size"] = pd.to_numeric(df["house_size"], errors='coerce')
df.drop_duplicates(ignore_index=True, inplace=True)

# Fix status
df['status'] = df['status'].replace(['for_salee','for_ssale'], 'for_sale')
df['status'] = df['status'].replace(['for_sale'], 1)
df['status'] = df['status'].replace(['ready_to_build'], 0)
df["status"] = pd.to_numeric(df["status"])
df = df[df['status'] > 0]

# Remove outliers
df['bed'] = df['bed'].replace([-2], 2)
df['acre_lot'] = df['acre_lot'].replace([100000, 99999, 96120], np.nan)
df['price'] = df['price'].replace([875000000, 0], np.nan)
df = df[df.price > 1000]
df = df[df.price != 169000000]
df = df[df.bath != 198]
df = df[df.bath != 123]
df = df[df.bed != 86]
df['house_size'] = df['house_size'].replace([-999], np.nan)
df = df[df.house_size != 1450112]
df = df[df.house_size != 400149]

# --- Automatic Categorical Encoding ---
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_cols:
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_features = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
    df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

# --- Impute missing house_size with Linear Regression ---
df_train = df.dropna(subset=['house_size'])
X_train_lr = df_train.drop('house_size', axis=1)
y_train_lr = df_train['house_size']

# Re-encode categorical columns for LR
categorical_cols_lr = X_train_lr.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_cols_lr:
    encoder_lr = OneHotEncoder(sparse=False, drop='first')
    encoded_features_lr = encoder_lr.fit_transform(X_train_lr[categorical_cols_lr])
    encoded_df_lr = pd.DataFrame(encoded_features_lr, columns=encoder_lr.get_feature_names_out(categorical_cols_lr))
    X_train_lr = pd.concat([X_train_lr.drop(columns=categorical_cols_lr), encoded_df_lr], axis=1)

lr = LinearRegression()
lr.fit(X_train_lr, y_train_lr)

# Predict missing house_size
df_missing = df[df['house_size'].isnull()]
if not df_missing.empty:
    X_missing = df_missing.drop('house_size', axis=1)
    if categorical_cols_lr:
        encoded_features_missing = encoder_lr.transform(X_missing[categorical_cols_lr])
        encoded_df_missing = pd.DataFrame(encoded_features_missing, columns=encoder_lr.get_feature_names_out(categorical_cols_lr))
        X_missing = pd.concat([X_missing.drop(columns=categorical_cols_lr), encoded_df_missing], axis=1)
    df_missing['house_size'] = lr.predict(X_missing)
    df.update(df_missing)

# --- Feature Scaling ---
numeric_cols = ['price', 'bed', 'bath', 'acre_lot', 'house_size']
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# --- Feature Selection ---
mi_scores = mutual_info_regression(df.drop('house_size', axis=1), df['house_size'])
mi_df = pd.DataFrame({'Feature': df.drop('house_size', axis=1).columns, 'MI Score': mi_scores})
mi_df = mi_df.sort_values(by='MI Score', ascending=False)
top_features = mi_df['Feature'][:10].tolist()
X = df[top_features]
y = df['house_size']

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Selection ---
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

cv_results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=kf)
    cv_results[name] = scores
    print(f"{name} 5-Fold CV R2: Mean={scores.mean():.4f}, Std={scores.std():.4f}")

# Choose best model (highest mean CV score)
best_model_name = max(cv_results, key=lambda k: cv_results[k].mean())
best_model = models[best_model_name]
best_model.fit(X_train, y_train)
print(f"\nBest model selected: {best_model_name}\n")

# --- Predictions ---
y_pred = best_model.predict(X_test)

# --- Error Analysis ---
residuals = y_test - y_pred
plt.figure(figsize=(8,5))
plt.scatter(y_test, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Actual House Size")
plt.ylabel("Residuals")
plt.title("Residual Analysis")
plt.show()

print(f"Test R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}\n")

# --- Model Interpretability ---
# SHAP for feature contributions
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)

# Partial Dependence Plots
PartialDependenceDisplay.from_estimator(best_model, X_train, features=top_features[:3], grid_resolution=50)
plt.show()

# --- Bias Detection ---
plt.figure(figsize=(10,4))
plt.hist(y_test, bins=50, alpha=0.5, label='Actual')
plt.hist(y_pred, bins=50, alpha=0.5, label='Predicted')
plt.xlabel('House Size')
plt.ylabel('Frequency')
plt.title('Distribution of Actual vs Predicted Values')
plt.legend()
plt.show()

bias = np.mean(y_pred - y_test)
print(f"Mean prediction bias: {bias:.4f}")

# --- Save Processed Data ---
df.to_csv('DataFrame_Cleaned_Processed.csv', index=False)
print("DataFrame final gravado em CSV!")
