import pandas as pd
import numpy as np
import logging
import mlflow
import mlflow.xgboost
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, 
                             confusion_matrix, 
                             roc_auc_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CLEANED_DATA_PATH = DATA_DIR / "Loan_Default_Cleaned.csv"
MODEL_DIR = BASE_DIR / "ml_engine" / "models"

logger.info(f"Data path: {CLEANED_DATA_PATH}")
logger.info(f"Model save path: {MODEL_DIR}")

def load_and_encode(path):
    """Load cleaned data and encode categorical columns"""
    logger.info("Loading cleaned dataset...")
    df = pd.read_csv(path)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # Identify categorical columns and encode them
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    logger.info(f"Encoding {len(categorical_cols)} categorical columns: {categorical_cols}")

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    logger.info("Encoding complete")
    return df

def prepare_features(df):
    """Split features/target and handle class imbalance with SMOTE"""
    
    # Drop ID and year — not useful for prediction
    drop_cols = ['ID', 'year', 'Interest_rate_spread', 'Upfront_charges', 'LTV']
    df = df.drop(columns=drop_cols)
    
    # Separate features and target
    X = df.drop(columns=['Status'])
    y = df['Status']
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution before SMOTE: {y.value_counts().to_dict()}")
    
    # Split into train and test BEFORE applying SMOTE
    # Important: never apply SMOTE to test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train size: {X_train.shape[0]} rows")
    logger.info(f"Test size: {X_test.shape[0]} rows")
    
    # Apply SMOTE only on training data
    logger.info("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    logger.info(f"After SMOTE - Target distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
    
    return X_train_balanced, X_test, y_train_balanced, y_test
def train_model(X_train, X_test, y_train, y_test):
    """Train XGBoost model with MLflow tracking"""
    
    logger.info("Starting model training with MLflow tracking...")
    
    mlflow.set_experiment("loan_default_prediction")
    
    with mlflow.start_run():
        # Define model parameters
        params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=100
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Log parameters and metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.xgboost.log_model(model, "xgboost_model")
        
        logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred))
        logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        
        # Save model to disk
        model_path = MODEL_DIR / "xgboost_loan_default.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to: {model_path}")
        
        return model, roc_auc

def explain_model(model, X_test):
    """Generate SHAP explanations for model predictions"""
    import shap
    
    logger.info("Generating SHAP explanations...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for test data
    # Using first 1000 rows only — full dataset takes too long
    X_sample = X_test.iloc[:1000]
    shap_values = explainer.shap_values(X_sample)
    
    # Get feature importance from SHAP
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 most important features:")
    print(feature_importance.head(10))
    
    # Save feature importance
    importance_path = MODEL_DIR / "feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved to: {importance_path}")
    
    return explainer, shap_values, X_sample

def explain_single_prediction(model, explainer, X_sample, index=0):
    """Explain one individual prediction in human readable format"""
    
    customer = X_sample.iloc[index]
    prediction = model.predict([customer])[0]
    probability = model.predict_proba([customer])[0][1]
    shap_vals = explainer.shap_values(customer.values.reshape(1, -1))[0]

    
    # Get top 3 reasons
    feature_shap = pd.DataFrame({
        'feature': X_sample.columns,
        'shap_value': shap_vals,
        'actual_value': customer.values
    }).sort_values('shap_value', ascending=False)
    
    print("\n" + "="*50)
    print(f"CUSTOMER {index} PREDICTION EXPLANATION")
    print("="*50)
    print(f"Prediction: {'DEFAULT' if prediction == 1 else 'NO DEFAULT'}")
    print(f"Default Probability: {probability:.1%}")
    print("\nTop reasons pushing toward DEFAULT:")
    print(feature_shap[feature_shap['shap_value'] > 0].head(3).to_string())
    print("\nTop reasons pushing toward NO DEFAULT:")
    print(feature_shap[feature_shap['shap_value'] < 0].tail(3).to_string())
    print("="*50)

if __name__ == "__main__":
    df = load_and_encode(CLEANED_DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_features(df)
    model, roc_auc = train_model(X_train, X_test, y_train, y_test)
    explainer, shap_values, X_sample = explain_model(model, X_test)
    explain_single_prediction(model, explainer, X_sample, index=0)