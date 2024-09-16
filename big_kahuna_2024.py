import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
from tqdm import tqdm
print('libraries imported!')

# Load and preprocess data
file_path = '/Users/patricksmith/Desktop/LinkedIn/2023 NCAA Tournament Study.xlsx'
sheet_name = 'Sheet7'
data = pd.read_excel(file_path, sheet_name=sheet_name)
print('File imported!')

X = data.iloc[:, 1:]  # Adjust based on your dataset structure
y = data.iloc[:, 0]   # Target variable

# Split the data
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)
print('Data split!')

# Preprocess the data
imputer = IterativeImputer(random_state=42)
print('Imputer called!')
X_train_imputed = imputer.fit_transform(X_train)
print('X_train data imputed!')
scaler = StandardScaler()
print('Scaler called!')
X_train_scaled = scaler.fit_transform(X_train_imputed)
print('X_train data scaled!')
X_imputed = imputer.fit_transform(X)  # Impute missing values
print('X data imputed!')
X_scaled = scaler.fit_transform(X_imputed)  # Scale the data
print('X data scaled!')

# Define PCA methods
pca_methods = {
    'Standard PCA': PCA(n_components=0.95),
    'Kernel PCA': KernelPCA(kernel='rbf', fit_inverse_transform=True, n_components=10),
    'Sparse PCA': SparsePCA(n_components=10, alpha=0.0001)
}
print('PCA methods initialized!')

# Define models with their initial parameter grid for tuning
models = {
    'Logistic Regression': (LogisticRegression(), {
        'model__C': np.logspace(-4, 4, 9),
        'model__solver': ['liblinear'],
        'model__penalty': ['l1', 'l2']
    }),
    'Decision Tree': (DecisionTreeClassifier(), {
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }),
    'Random Forest': (RandomForestClassifier(), {
        'model__n_estimators': [100, 200, 500],
        'model__max_features': ['auto', 'sqrt', 'log2'],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }),
    'Gradient Boosting': (GradientBoostingClassifier(), {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.001, 0.01, 0.1, 1],
        'model__max_depth': [3, 5, 10]
    }),
    'XGBoost': (XGBClassifier(), {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.3],
        'model__max_depth': [3, 6, 10],
        'model__colsample_bytree': [0.3, 0.7, 1]
    }),
    'SVM': (SVC(probability=True), {
        'model__C': np.logspace(-2, 2, 5),
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto']
    })
}
print('Models initialized with initial hyperparameters!')

# Define scoring metrics
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
print('scoring metrics initialized!')

# Initialize nested cross-validation
outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
print('outer_cv initialized!')

results = []
print('Empty results array created!')

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
print('Warnings suppressed!')

# For each PCA method and model combination
for pca_name, pca in tqdm(pca_methods.items(), desc='PCA Methods'):
    for model_name, (model, param_grid) in tqdm(models.items(), desc='Models', leave=False):
        # Create pipeline
        pipeline = Pipeline([('pca', pca), ('model', model)])

        # Inner cross-validation for hyperparameter tuning
        inner_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=42)
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring='accuracy',
                                   refit=True)

        # Evaluate the model with GridSearchCV within the outer cross-validation loop
        for train_ix, test_ix in tqdm(outer_cv.split(X_scaled, y), desc='Outer CV', leave=False):
            X_train, X_test = X_scaled[train_ix], X_scaled[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]

            # Fit the grid search and get the best model
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            # Evaluate the best model on the held-out test set
            test_scores = cross_validate(best_model, X_test, y_test, scoring=scoring_metrics)

            # Store results
            mean_scores = {metric: np.mean(scores) for metric, scores in test_scores.items() if 'test_' in metric}
            results.append({
                'PCA': pca_name,
                'Model': model_name,
                'Best Parameters': grid_search.best_params_,
                **mean_scores
            })

# Convert results to DataFrame, sort by accuracy, and export
results_df = pd.DataFrame(results).sort_values(by='test_accuracy', ascending=False)
print('results dataframe created!')
print(results_df)
results_df.to_excel('nested_cv_model_optimization_results.xlsx', index=False)
