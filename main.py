# import Libraries
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.exceptions import UndefinedMetricWarning

# ignore warnings
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

# load Dataset
df = pd.read_csv('./insurance.csv')
print("Dataset shape:", df.shape, "\n\n")
print(df.head(), "\n\n")

# cat/num features
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']

# --- data understanding ---

# Basic statistics
print(df.describe(), "\n\n")
print(df.info(), "\n\n")
print(df.isnull().sum(), "\n\n")

# check categorical variables
for col in categorical_features:
    print(f"----- Value counts for {col} -----")
    print(df[col].value_counts())
    print("\n\n")

# outlier detection
for col in numerical_features:
    if col in ['age']:  # treat as normal distribution
        mean = df[col].mean()
        std = df[col].std()
        upper = mean + 3*std
        lower = mean - 3*std
        outliers = df[(df[col] > upper) | (df[col] < lower)]
        print(f"No. of outliers in '{col}' (mean ± 3*std): {outliers.shape[0]}")
    else:  # treat as skewed distribution
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"No. of outliers in '{col}' (IQR method): {outliers.shape[0]}")
print("\n\n")

# visualize
plt.figure(figsize=(16,8))
for i, col in enumerate(numerical_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(
        y=df[col], 
        color='lightblue',  
        flierprops=dict(marker='o', markerfacecolor='red', markersize=6, linestyle='none') 
    )
    plt.title(f"Boxplot of {col}", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('results/1_numerical_outliers.png')


# histogram of charges
plt.figure(figsize=(8,6))
sns.histplot(df['charges'], bins=30, kde=True)
plt.title("Distribution of Insurance Charges")
plt.savefig('results/2_charges_distribution.png')
plt.close() 

# countplot for smoker
plt.figure(figsize=(6,4))
sns.countplot(x='smoker', data=df)
plt.title("Smoker vs Non-smoker")
plt.savefig('results/3_smoker_count.png')
plt.close()

# boxplot of charges by smoker
plt.figure(figsize=(6,4))
sns.boxplot(x='smoker', y='charges', data=df)
plt.title("Charges by Smoker Status")
plt.savefig('results/4_charges_by_smoker.png')
plt.close()

# pairplot colored by smoker
sns.pairplot(df, hue='smoker')
plt.savefig('results/5_pairplot_by_smoker.png')
plt.close()

# --- data Preparation ---

# handle missing values
for col in numerical_features:
    if df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
        print(f"Filled missing values in '{col}' with median value {median_value}\n\n")
print(df.isnull().sum(), "\n\n")

# --- outlier handling ---

# IQR
Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR
print(f"BMI lower bound: {lower_bound}, upper bound: {upper_bound}")

# cap and transform
df['bmi_capped'] = df['bmi'].clip(lower=lower_bound, upper=upper_bound)
df['bmi_sqrt'] = np.sqrt(df['bmi'])
print('Original BMI min/max:', df['bmi'].min(), df['bmi'].max())
print('Capped BMI min/max:', df['bmi_capped'].min(), df['bmi_capped'].max())
print('Sqrt BMI min/max:', df['bmi_sqrt'].min(), df['bmi_sqrt'].max(), "\n\n")

# visualize
plt.figure(figsize=(18,5))
plt.subplot(1,3,1)
sns.histplot(df['bmi'], bins=30, kde=True, color='skyblue')
plt.title("BMI Original")
plt.subplot(1,3,2)
sns.histplot(df['bmi_capped'], bins=30, kde=True, color='lightgreen')
plt.title("BMI Capped")
plt.subplot(1,3,3)
sns.histplot(df['bmi_sqrt'], bins=30, kde=True, color='salmon')
plt.title("BMI Sqrt-transformed")
plt.tight_layout()
plt.savefig('results/6_bmi_comparison_histogram.png')

# choose capped/transformed 
df['bmi'] = df['bmi_capped'] 
df.drop(['bmi_capped', 'bmi_sqrt'], axis=1, inplace=True)

# standardize and encode
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# train-test split
X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- data modeling ---

# models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# results
results = {
    'holdout': {},
    'kfold': {},
    'bootstrap': {}
}

# Hold-out sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
for name, model in models.items():
    pipeline = Pipeline([('preprocessor', preprocessor),
                         ('regressor', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results['holdout'][name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

# k-Fold Cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
for name, model in models.items():
    pipeline = Pipeline([('preprocessor', preprocessor),
                         ('regressor', model)])
    r2_scores = cross_val_score(pipeline, X, y, scoring=make_scorer(r2_score), cv=kf)
    mse_scores = cross_val_score(pipeline, X, y, scoring='neg_mean_squared_error', cv=kf)
    rmse_scores = np.sqrt(-mse_scores)
    mae_scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=kf) * -1
    results['kfold'][name] = {
        'Mean_R2': np.mean(r2_scores),
        'Mean_MSE': np.mean(-mse_scores),
        'Mean_RMSE': np.mean(rmse_scores),
        'Mean_MAE': np.mean(mae_scores)
    }

# Bootstrapping
n_iterations = 10
for name, model in models.items():
    r2_list, mse_list, rmse_list, mae_list = [], [], [], []
    for i in range(n_iterations):
        X_train_bs, X_test_bs, y_train_bs, y_test_bs = train_test_split(
            X, y, test_size=0.2, random_state=42+i)
        pipeline = Pipeline([('preprocessor', preprocessor),
                             ('regressor', model)])
        pipeline.fit(X_train_bs, y_train_bs)
        y_pred_bs = pipeline.predict(X_test_bs)
        mse_bs = mean_squared_error(y_test_bs, y_pred_bs)
        r2_list.append(r2_score(y_test_bs, y_pred_bs))
        mse_list.append(mse_bs)
        rmse_list.append(np.sqrt(mse_bs))
        mae_list.append(mean_absolute_error(y_test_bs, y_pred_bs))
    results['bootstrap'][name] = {
        'Mean_R2': np.mean(r2_list),
        'Mean_MSE': np.mean(mse_list),
        'Mean_RMSE': np.mean(rmse_list),
        'Mean_MAE': np.mean(mae_list)
    }

# print results
print("=== Evaluation Results ===\n")
for method, res in results.items():
    print(f"{method.upper()} results:")
    for name, metrics in res.items():
        print(f"{name}: {metrics}")
    print("\n")

# choose best model based on R2
w_r2, w_rmse, w_mae = 0.4, 0.3, 0.3

best_method = None
best_model = None
best_score = -np.inf

for method, res in results.items():
    for name, metrics in res.items():
        r2 = metrics.get('R2', metrics.get('Mean_R2'))
        mse = metrics.get('MSE', metrics.get('Mean_MSE'))
        rmse = metrics.get('RMSE', np.sqrt(mse))
        mae = metrics.get('MAE', metrics.get('Mean_MAE'))

        rmse_score = 1 / (1 + rmse)
        mae_score = 1 / (1 + mae)
        combined = w_r2*r2 + w_rmse*rmse_score + w_mae*mae_score

        if combined > best_score:
            best_score = combined
            best_method = method
            best_model = name
print("Best Model:", best_model)
print("Best Sampling Method:", best_method)
print("Best Combined Score:", round(best_score, 3))

# visualie results
residuals = y_test - y_pred

# Scatter plot: Predicted vs Actual
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/7_predicted_vs_actual.png')

# Residual plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_pred, y=residuals, color='green', alpha=0.6)
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/8_residual_plot.png')

# Histogram of residuals
plt.figure(figsize=(8,6))
sns.histplot(residuals, bins=30, kde=True, color='purple')
plt.xlabel('Residuals')
plt.title('Residuals Distribution')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/9_residuals_hist.png')