import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2

# region ========================================= Load Data =======================================================
df = pd.read_csv(zipfile.ZipFile(r'C:\Users\drory\learning-project\Data Science exercises\Projects\Random_Forest_&_XGboost\Diabetes\Data\Diabetes Data.zip').open('diabetes.csv'))

# region ========================================= Data Cleaning ==================================================
# df.replace(0, np.nan, inplace=True)
# df.fillna(df.median(), inplace=True)
#


# Provides a statistical summary of the DataFrame for numerical columns (e.g., mean, std, min, max)
print(df.describe().transpose())

# Displays an overview of the DataFrame, including column data types, non-null counts, and memory usage
print(df.info())

df.hist(bins=50, figsize=(12, 8))

# Replace all zero values in the 'SkinThickness' column with NaN, as 0 likely represents missing data
df['SkinThickness'].replace(0, np.nan, inplace=True)
df['Insulin'].replace(0, np.nan, inplace=True)

# Store valid (non-NaN) values from the 'SkinThickness' column for random sampling
valid_values_SkinThickness = df['SkinThickness'].dropna()
valid_values_Insulin = df['Insulin'].dropna()


# Fill missing (NaN) values by randomly sampling from the existing (valid) distribution of 'SkinThickness'
# Using `.astype(int)` ensures that the data type remains consistent (integer) after replacement
df.loc[df['SkinThickness'].isnull(), 'SkinThickness'] = np.random.choice(valid_values_SkinThickness, size=df['SkinThickness'].isnull().sum(), replace=True).astype(int)
df.loc[df['Insulin'].isnull(), 'Insulin'] = np.random.choice(valid_values_Insulin, size=df['Insulin'].isnull().sum(), replace=True).astype(int)


df['BloodPressure'].replace(0, np.nan, inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].median(), inplace=True)

df['BMI'].replace(0, np.nan, inplace=True)
df['BMI'].fillna(df['BMI'].median(), inplace=True)


df = df[df['Glucose'] != 0]
df = df[(df[['Glucose', 'BloodPressure', 'BMI', 'Insulin']] != 0).all(axis=1)]
print(df.info())

df.hist(bins=50, figsize=(12, 8))
# region ========================================= Exploratory Data Analysis ======================================
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='Blues', fmt='.2f')
plt.show()

# KDE plot - איך משתנים מתפלגים לפי `Outcome`
plt.figure(figsize=(12, 6))
for col in ['Glucose', 'BMI', 'Insulin', 'Age']:
    sns.kdeplot(df[col][df['Outcome'] == 1], label=f"{col} (Diabetic)", shade=True)
    sns.kdeplot(df[col][df['Outcome'] == 0], label=f"{col} (Non-Diabetic)", shade=True)
plt.legend()
plt.title("Distribution of Variables by Outcome")
plt.show()

# region ===================================== Feature Selection ================================================
k = 5  # מספר הפיצ'רים שנבחרו
selector = SelectKBest(score_func=chi2, k=k)
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("Selected Features:", selected_features.tolist())

# region ===================================== Train/Test Split and Scaling ======================================
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.20, random_state=41)
# scaler = MinMaxScaler().fit(X_train)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# region ===================================== Hyperparameter Tuning for XGBoost ================================
from sklearn.model_selection import GridSearchCV

# הגדרת הפרמטרים שברצוננו לבדוק
param_grid = {
    'max_depth': [3, 6, 9, 11, 15, 21],  # עומק העצים, משפיע על מורכבות המודל
    'learning_rate': [0.01, 0.1, 0.3],  # קצב הלמידה, מאזן בין התכנסות מהירה ליציבות
    'n_estimators': [10, 20, 30, 50, 100, 200],  # מספר העצים ביער החיזוי
    'reg_alpha': [0, 0.1, 0.3],  # רגולריזציה L1 להפחתת Overfitting
    'reg_lambda': [1, 1.5, 2]  # רגולריזציה L2 להפחתת Overfitting
}

xgb = XGBClassifier()
grid_search = GridSearchCV(xgb, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

# הרצת החיפוש על סט האימון
print("Running GridSearchCV...")
grid_search.fit(X_train, y_train)

# הצגת הפרמטרים הטובים ביותר
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# region ===================================== Model Evaluation ================================================
y_pred_test = grid_search.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print("Final Model Accuracy on Test Set: %.2f%%" % (accuracy * 100.0))

cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Diabetic', 'Diabetic'])
disp.plot()
plt.show()
