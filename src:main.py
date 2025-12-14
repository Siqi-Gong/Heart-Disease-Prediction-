# Heart Disease Classification Pipeline
# Key Focus: Solving Class Imbalance & Maximizing Recall

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

# Import utility functions
from utils import load_data, build_preprocessor, evaluate_model

# Constants / Settings
DATA_URL = 'https://raw.githubusercontent.com/Bubu631/Heart-Disease-Classification-Imbalanced/refs/heads/main/heart_cleaned.csv'
RANDOM_STATE = 42

def main():
    # 1. Load Data
    df = load_data(DATA_URL)
    
    # 2. Data Preparation
    # Convert target variable 'HeartDisease' to binary (0/1)
    df['HeartDisease'] = df['HeartDisease'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']
    
    # Calculate Imbalance Ratio for XGBoost (scale_pos_weight)
    # Ratio = Count(Negative) / Count(Positive)
    neg, pos = np.bincount(y)
    scale_pos_weight = neg / pos
    print(f"[Info] Imbalance Ratio (Neg/Pos): {scale_pos_weight:.2f}")

    # 3. Split Dataset
    # Stratified sampling is ensured by `stratify=y` to maintain class distribution
    print("[Preprocessing] Splitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Get the preprocessing pipeline
    preprocessor = build_preprocessor()

    # ==========================================
    # Model 1: Logistic Regression (Baseline)
    # Strategy: Use `class_weight='balanced'` to handle imbalance
    # ==========================================
    print("\n>>> Training Logistic Regression (Weighted)...")
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', solver='liblinear', random_state=RANDOM_STATE))
    ])
    lr_pipeline.fit(X_train, y_train)
    evaluate_model(lr_pipeline, X_test, y_test, "Logistic Regression (Weighted)")

    # ==========================================
    # Model 2: Random Forest
    # Strategy: `class_weight='balanced'` and `max_depth` to prevent overfitting
    # ==========================================
    print("\n>>> Training Random Forest (Weighted)...")
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight='balanced', max_depth=10, random_state=RANDOM_STATE))
    ])
    rf_pipeline.fit(X_train, y_train)
    evaluate_model(rf_pipeline, X_test, y_test, "Random Forest")

    # ==========================================
    # Model 3: XGBoost (Advanced Tuning)
    # Strategy: `scale_pos_weight` and GridSearchCV with StratifiedKFold
    # ==========================================
    print("\n>>> Tuning XGBoost with Grid Search...")
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'))
    ])
    
    # Define hyperparameter search space
    param_grid = {
        'classifier__scale_pos_weight': [scale_pos_weight], # Crucial for imbalance
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__n_estimators': [100, 200]
    }
    
    # 5-fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Optimize for Recall (Sensitivity)
    grid_search = GridSearchCV(
        xgb_pipeline, 
        param_grid, 
        cv=cv, 
        scoring='recall', # Target Metric
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"[XGBoost] Best Hyperparameters: {grid_search.best_params_}")
    best_xgb = grid_search.best_estimator_
    evaluate_model(best_xgb, X_test, y_test, "XGBoost (Tuned)")

if __name__ == "__main__":
    main()
