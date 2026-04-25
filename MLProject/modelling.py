"""
modelling.py — MLProject version
Nama   : Fauzan Aidil Luthfi
NIM    : apc352d6y0439

"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, log_loss
)


# ============================================================
# CONFIG
# ============================================================
DATA_PATH    = 'titanic_preprocessing/train_preprocessed.csv'
ARTIFACT_DIR = 'artifacts'
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================
def load_data(path):
    df = pd.read_csv(path)
    X  = df.drop(columns=['Survived'])
    y  = df['Survived']
    print(f'Data dimuat — Shape: {df.shape}')
    return X, y, X.columns.tolist()


# ============================================================
# ARTEFAK: Confusion Matrix
# ============================================================
def save_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Tidak Selamat', 'Selamat'],
        yticklabels=['Tidak Selamat', 'Selamat']
    )
    plt.title('Confusion Matrix — Titanic')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'Confusion matrix disimpan: {path}')


# ============================================================
# ARTEFAK: Feature Importance
# ============================================================
def save_feature_importance(model, feature_names, path):
    fi_df = pd.DataFrame({
        'Feature':    feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'Feature importance disimpan: {path}')


# ============================================================
# MAIN
# ============================================================
def train():
    mlflow.set_experiment('Titanic_CI_Workflow')

    X, y, feature_names = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'Train: {len(X_train)} | Test: {len(X_test)}')

    with mlflow.start_run():

        # Train
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cv     = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

        # Manual logging params
        mlflow.log_param('model_type',        'RandomForestClassifier')
        mlflow.log_param('n_estimators',      200)
        mlflow.log_param('max_depth',         6)
        mlflow.log_param('min_samples_split', 5)
        mlflow.log_param('random_state',      42)

        # Manual logging metrics
        mlflow.log_metric('accuracy',         accuracy_score(y_test, y_pred))
        mlflow.log_metric('precision',        precision_score(y_test, y_pred))
        mlflow.log_metric('recall',           recall_score(y_test, y_pred))
        mlflow.log_metric('f1_score',         f1_score(y_test, y_pred))
        mlflow.log_metric('roc_auc',          roc_auc_score(y_test, y_prob))
        mlflow.log_metric('log_loss',         log_loss(y_test, y_prob))
        mlflow.log_metric('cv_mean_accuracy', cv.mean())
        mlflow.log_metric('cv_std_accuracy',  cv.std())

        # Log model
        mlflow.sklearn.log_model(model, artifact_path='model')

        # Artefak tambahan
        cm_path = f'{ARTIFACT_DIR}/confusion_matrix.png'
        save_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path, artifact_path='plots')

        fi_path = f'{ARTIFACT_DIR}/feature_importance.png'
        save_feature_importance(model, feature_names, fi_path)
        mlflow.log_artifact(fi_path, artifact_path='plots')

        report = classification_report(y_test, y_pred, output_dict=True)
        cr_path = f'{ARTIFACT_DIR}/classification_report.json'
        with open(cr_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(cr_path, artifact_path='reports')

        print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
        print(f'AUC     : {roc_auc_score(y_test, y_prob):.4f}')
        print('Training selesai!')


if __name__ == '__main__':
    train()
