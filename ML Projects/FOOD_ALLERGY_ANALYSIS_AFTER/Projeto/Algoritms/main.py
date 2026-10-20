import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from imblearn.over_sampling import SMOTE   # ADD for oversampling minority class
from functions import resolve_outliers
from sklearn.ensemble import RandomForestClassifier
import joblib
import shap

df = pd.read_csv('./food-allergy-analysis-Zenodo.csv', low_memory=False).replace('fALSE', 'FALSE').replace(np.NaN, 0) \
    .replace("'NA'", 0).replace('na', 0).replace('\'', '')
df = df.drop(df.columns[0], axis=1).drop_duplicates(ignore_index=True)

le = LabelEncoder()
df['GENDER_FACTOR_ENCODER'] = le.fit_transform(df['GENDER_FACTOR'])
df['RACE_FACTOR_ENCODER'] = le.fit_transform(df['RACE_FACTOR'])
df['ETHNICITY_FACTOR_ENCODER'] = le.fit_transform(df['ETHNICITY_FACTOR'])
df['PAYER_FACTOR_ENCODER'] = le.fit_transform(df['PAYER_FACTOR'])
df['ATOPIC_MARCH_COHORT_ENCODER'] = le.fit_transform(df['ATOPIC_MARCH_COHORT'])

for line in range(0, len(df)):
    if df['ATOPIC_MARCH_COHORT_ENCODER'][line] == 2:
        df.loc[line, 'ATOPIC_MARCH_COHORT_ENCODER'] = 0
    if df['RACE_FACTOR_ENCODER'][line] == 4:
        df.loc[line, 'RACE_FACTOR_ENCODER'] = 3
    if type(df['BIRTH_YEAR'][line]) == str:
        df.loc[line, 'BIRTH_YEAR'] = pd.to_numeric(df['BIRTH_YEAR'][line].replace('\'', ''), downcast="integer")
    if type(df['AGE_START_YEARS'][line]) == str:
        df.loc[line, 'AGE_START_YEARS'] = pd.to_numeric(df['AGE_START_YEARS'][line].replace('\'', ''))
    if type(df['MILK_ALG_START'][line]) == str:
        df.loc[line, 'MILK_ALG_START'] = pd.to_numeric(df['MILK_ALG_START'][line].replace('\'', ''), downcast="float")
    if type(df['WHEAT_ALG_END'][line]) == str:
        df.loc[line, 'WHEAT_ALG_END'] = pd.to_numeric(df['WHEAT_ALG_END'][line].replace('\'', ''), downcast="float")
    if type(df['PEANUT_ALG_END'][line]) == str:
        df.loc[line, 'PEANUT_ALG_END'] = pd.to_numeric(df['PEANUT_ALG_END'][line].replace('\'', ''), downcast="float")
    if df['SHELLFISH_ALG_END'][line] > 0 or df['FISH_ALG_END'][line] > 0 or df['MILK_ALG_END'][line] > 0 or \
            df['SOY_ALG_END'][line] > 0 or df['EGG_ALG_END'][line] > 0 or df['WHEAT_ALG_END'][line] > 0 or \
            df['PEANUT_ALG_END'][line] > 0 or df['SESAME_ALG_END'][line] > 0 or df['TREENUT_ALG_END'][line] > 0 or \
            df['WALNUT_ALG_END'][line] > 0 or df['PECAN_ALG_END'][line] > 0 or df['PISTACH_ALG_END'][line] > 0 or \
            df['ALMOND_ALG_END'][line] > 0 or df['BRAZIL_ALG_END'][line] > 0 or df['HAZELNUT_ALG_END'][line] > 0 or \
            df['CASHEW_ALG_END'][line] > 0 or df['ATOPIC_DERM_END'][line] > 0 or df['ALLERGIC_RHINITIS_END'][line] > 0 \
            or df['ASTHMA_END'][line] > 0:
        df.loc[line, 'HAS_ALLERGIES'] = 1
    else:
        df.loc[line, 'HAS_ALLERGIES'] = 0

print(df.describe())
plt.clf()
# AGE_END_YEARS
plt.subplot(1, 2, 1)  # before
plt.boxplot(df['AGE_END_YEARS'])
plt.title("Original")
#
df['AGE_END_YEARS'] = resolve_outliers(df['AGE_END_YEARS'])  # after
plt.subplot(1, 2, 2)
plt.boxplot(df['AGE_END_YEARS'])
plt.title("After")

# AGE_START_YEARS
plt.clf()
plt.subplot(1, 2, 1)  # before
plt.boxplot(df['AGE_START_YEARS'])
plt.title("Original")

df['AGE_START_YEARS'] = resolve_outliers(df['AGE_START_YEARS'])  # after
plt.subplot(1, 2, 2)
plt.boxplot(df['AGE_END_YEARS'])
plt.title("After")

# BIRTH_YEAR
plt.clf()
plt.subplot(1, 2, 1)  # before
plt.boxplot(df['BIRTH_YEAR'])
plt.title("Original")
#
df['BIRTH_YEAR'] = resolve_outliers(df['BIRTH_YEAR'])  # after
plt.subplot(1, 2, 2)
plt.boxplot(df['BIRTH_YEAR'])
plt.title("After")

threshold = 10
independent_variables = ['BIRTH_YEAR', 'GENDER_FACTOR_ENCODER', 'RACE_FACTOR_ENCODER', 'ETHNICITY_FACTOR_ENCODER',
                         'PAYER_FACTOR_ENCODER', 'ATOPIC_MARCH_COHORT_ENCODER', 'AGE_START_YEARS', 'AGE_END_YEARS',
                         'SHELLFISH_ALG_START', 'SHELLFISH_ALG_END', 'FISH_ALG_START', 'FISH_ALG_END', 'MILK_ALG_START',
                         'MILK_ALG_END', 'SOY_ALG_START', 'SOY_ALG_END', 'EGG_ALG_START', 'EGG_ALG_END',
                         'WHEAT_ALG_START', 'WHEAT_ALG_END', 'PEANUT_ALG_START', 'PEANUT_ALG_END', 'SESAME_ALG_START',
                         'SESAME_ALG_END', 'WALNUT_ALG_START', 'WALNUT_ALG_END', 'ALMOND_ALG_START',
                         'ALMOND_ALG_END', 'CASHEW_ALG_START', 'CASHEW_ALG_END', 'ATOPIC_DERM_START',
                         'ATOPIC_DERM_END', 'ALLERGIC_RHINITIS_START', 'ALLERGIC_RHINITIS_END', 'ASTHMA_START',
                         'ASTHMA_END']

x = df[independent_variables].astype('float')
for i in np.arange(0, len(independent_variables)):
    vif = [variance_inflation_factor(x[independent_variables], ix) for ix in range(x[independent_variables].shape[1])]
    maxloc = vif.index(max(vif))
    if max(vif) > threshold:
        print('dropping ', x[independent_variables].columns[maxloc], ' at index ', maxloc)
        del independent_variables[maxloc]
    else:
        break

scaler = StandardScaler()
scaler.fit(x)
xpca = scaler.transform(x)

pca = PCA(.99)
pca.fit(xpca)

x = pca.transform(xpca)
y = df['HAS_ALLERGIES']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, stratify=y)  # stratify keeps class balance

# ---------------------- HYPERPARAMETER TUNING ----------------------
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': [None, 'balanced']   # ADD cost-sensitive option
}

rf = RandomForestClassifier(random_state=42)

# Option 1: Grid Search
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=3,
                           scoring='f1', # switch to F1 (better for imbalanced)
                           n_jobs=-1,
                           verbose=2)

# Option 2: Randomized Search
random_search = RandomizedSearchCV(estimator=rf,
                                   param_distributions=param_grid,
                                   n_iter=10,
                                   cv=3,
                                   scoring='accuracy',
                                   n_jobs=-1,
                                   verbose=2,
                                   random_state=42)

USE_RANDOM_SEARCH = False  # Change to True for RandomizedSearchCV

if USE_RANDOM_SEARCH:
    random_search.fit(x_train, y_train)
    model = random_search.best_estimator_
    print("Best Hyperparameters (RandomizedSearchCV):", random_search.best_params_)
    print("Best CV Accuracy:", random_search.best_score_)
else:
    grid_search.fit(x_train, y_train)
    model = grid_search.best_estimator_
    print("Best Hyperparameters (GridSearchCV):", grid_search.best_params_)
    print("Best CV Accuracy:", grid_search.best_score_)
# -------------------------------------------------------------------

# ---------------------- K-FOLD CROSS-VALIDATION ----------------------
print("\nPerforming 5-Fold Cross-Validation on the final model...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, x, y, cv=kf, scoring='accuracy', n_jobs=-1)
print("Cross-validation scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))
print("Std Dev of CV Accuracy:", np.std(cv_scores))
# ---------------------------------------------------------------------

# ---------------------- Model Interpretability with SHAP ----------------------
print("\nGenerating SHAP explanations for the Random Forest model...")

# Create an explainer using TreeExplainer for RandomForest
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_test)

# For binary classification, shap_values[1] corresponds to positive class
# Summary plot (global feature importance)
shap.summary_plot(shap_values[1], x_test, feature_names=independent_variables, show=False)
plt.savefig('Images/SHAP_summary_plot.png')
plt.close()

# Optional: force plot for first few test observations (local explanation)
for i in range(5):
    shap.force_plot(explainer.expected_value[1], shap_values[1][i], x_test[i], feature_names=independent_variables, matplotlib=True)
    plt.savefig(f'Images/SHAP_force_plot_{i}.png')
    plt.close()

print("SHAP plots saved successfully!")
# --------------------------------------------------------------------------

# Normal test set evaluation
predictions = model.predict(x_test)

numberObservations = 50
x_range = range(len(y_test[:numberObservations]))
plt.clf()
plt.figure(figsize=(20, 10))
plt.plot(x_range, y_test[:numberObservations], label='True')
plt.plot(x_range, predictions[:numberObservations], label='Predicted')
plt.title('Indicativo de Alergias')
plt.xlabel('Observações')
plt.ylabel('Indicativo')
plt.xticks(np.arange(numberObservations))
plt.legend()
plt.savefig('Images/ModelPredicts')
plt.show()

r2Score = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("R2 no test: ", r2Score)
print("Mean absolute error: ", round(mse, 2))
print("Mean squared logarithmic error:", round(rmse, 2))
print("R2 durante o treino   :", round(r2_score(y_train, model.predict(x_train)), 2))

print(classification_report(y_test, predictions))

joblib.dump(model, '../Frontend/model/random_forest_model.sav')
