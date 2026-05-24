# -*- coding: utf-8 -*-
"""
classification.py
Classification Modeling script for Hotel Booking Demand Project
"""

# Import libraries for classification modeling
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def main():
    warnings.filterwarnings('ignore')
    plt.style.use('default')
    sns.set_theme(style='whitegrid')
    RANDOM_STATE = 42

    # Directory setup
    FIGURE_DIR = Path('../reports/figures')
    if not FIGURE_DIR.exists():
        FIGURE_DIR = Path('reports/figures')
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


    # Load the processed classification data
    data_path = '../data/processed/hotel_bookings_clf.csv'
    if not os.path.exists(data_path):
        data_path = 'data/processed/hotel_bookings_clf.csv'

    df = pd.read_csv(data_path)

    print('Data path:', data_path)
    print('Shape:', df.shape)


    # Check target balance and basic data quality
    print('\nMissing values:', int(df.isnull().sum().sum()))
    print('Object columns:', len(df.select_dtypes(include='object').columns))
    print('\nTarget count:')
    print(df['is_canceled'].value_counts().sort_index())
    print('\nTarget ratio:')
    print((df['is_canceled'].value_counts(normalize=True).sort_index() * 100).round(2))


    # Feature and Target Setup
    target = 'is_canceled'
    assigned_cols = [col for col in df.columns if col.startswith('assigned_room_type_')]
    drop_cols = [target, 'adr'] + assigned_cols

    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target]

    print('\nTarget:', target)
    print('Dropped columns:', len(drop_cols))
    print('Assigned room columns removed:', len(assigned_cols))
    print('Feature shape:', X.shape)
    print('Target shape:', y.shape)


    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print('\nTrain shape:', X_train.shape)
    print('Test shape:', X_test.shape)
    print('\nTrain target ratio:')
    print((y_train.value_counts(normalize=True).sort_index() * 100).round(2))
    print('\nTest target ratio:')
    print((y_test.value_counts(normalize=True).sort_index() * 100).round(2))


    # Model Candidates
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(
            n_estimators=50,
            max_depth=12,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ['accuracy', 'precision', 'recall', 'f1']

    print('\nModels:', list(models.keys()))
    print('CV:', cv)


    # Compare models with 5-fold cross validation
    cv_rows = []
    print("\nRunning Cross Validation...")
    for name, model in models.items():
        print('Running:', name)
        score = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False
        )
        cv_rows.append({
            'model': name,
            'cv_accuracy': score['test_accuracy'].mean(),
            'cv_precision': score['test_precision'].mean(),
            'cv_recall': score['test_recall'].mean(),
            'cv_f1': score['test_f1'].mean()
        })

    cv_result = pd.DataFrame(cv_rows).sort_values('cv_f1', ascending=False)
    print('\n[Cross Validation Results]')
    print(cv_result.round(4).to_string(index=False))


    # Model Comparison Visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = cv_result.melt(id_vars='model', value_vars=['cv_accuracy', 'cv_precision', 'cv_recall', 'cv_f1'])
    sns.barplot(data=plot_df, x='model', y='value', hue='variable', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title('5-Fold Cross Validation Model Comparison')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.legend(title='Metric', loc='lower right')
    plt.xticks(rotation=15)
    plt.tight_layout()
    comparison_path = FIGURE_DIR / 'classification_model_comparison.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('\nSaved figure:', comparison_path)


    # Hyperparameter Tuning via GridSearchCV
    best_name = cv_result.iloc[0]['model']
    print(f"\nInitial Best Model for Tuning: {best_name}")

    if best_name == 'Decision Tree':
        param_grid = {
            'max_depth': [8, 10, 12, 15],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
        tuning_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    elif best_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 12, 15],
            'min_samples_split': [2, 5]
        }
        tuning_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    else:
        param_grid = {
            'model__C': [0.1, 1.0, 10.0]
        }
        tuning_model = models['Logistic Regression']

    print(f"GridSearchCV parameter grid:\n{param_grid}")

    grid_search = GridSearchCV(
        estimator=tuning_model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters found by GridSearchCV:", grid_search.best_params_)
    print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

    models[best_name] = grid_search.best_estimator_


    # Final Test Evaluation
    best_name = cv_result.iloc[0]['model']
    best_model = models[best_name]
    safe_name = best_name.lower().replace(' ', '_')

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    test_result = pd.DataFrame([{
        'model': best_name,
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
        'test_f1': f1_score(y_test, y_pred)
    }])

    print('\nBest model after tuning:', best_name)
    print(test_result.round(4).to_string(index=False))
    
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, target_names=['Not canceled', 'Canceled']))

    cm = confusion_matrix(y_test, y_pred)
    cm_table = pd.DataFrame(
        cm,
        index=['Actual not canceled', 'Actual canceled'],
        columns=['Predicted not canceled', 'Predicted canceled']
    )

    print('\nConfusion matrix:')
    print(cm_table.to_string())

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not canceled', 'Canceled'])
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    ax.set_title(f'Confusion Matrix: {best_name}')
    plt.tight_layout()
    confusion_path = FIGURE_DIR / f'confusion_matrix_{safe_name}.png'
    plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved figure:', confusion_path)


    # Feature Interpretation
    model_part = best_model.named_steps['model'] if hasattr(best_model, 'named_steps') else best_model

    if hasattr(model_part, 'feature_importances_'):
        importance = model_part.feature_importances_
        title = 'Feature Importance'
    elif hasattr(model_part, 'coef_'):
        importance = np.abs(model_part.coef_[0])
        title = 'Absolute Coefficient'
    else:
        importance = np.zeros(len(X.columns))
        title = 'Importance'

    imp_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    }).sort_values('importance', ascending=False).head(15)

    print('\n[Top 15 Features]')
    print(imp_df.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=imp_df, x='importance', y='feature', ax=ax, color='#4C72B0')
    ax.set_title(f'Top 15 Features: {best_name}')
    ax.set_xlabel(title)
    ax.set_ylabel('Feature')
    plt.tight_layout()
    importance_path = FIGURE_DIR / 'classification_feature_importance.png'
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved figure:', importance_path)

    # Summarize the final classification result
    print('\n' + '='*50)
    print('Final selected model:', best_name)
    print('Number of input features:', X.shape[1])
    print('Excluded adr:', 'adr' in drop_cols)
    print('Excluded assigned_room_type columns:', len(assigned_cols))
    print('='*50)

if __name__ == "__main__":
    main()