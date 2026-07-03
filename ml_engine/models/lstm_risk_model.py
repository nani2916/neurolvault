import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout,
                                      LayerNormalization,
                                      MultiHeadAttention,
                                      Flatten, Reshape)
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "ml_engine" / "models"
CLEANED_DATA_PATH = DATA_DIR / "Loan_Default_Cleaned.csv"

logger.info(f"Data path: {CLEANED_DATA_PATH}")
logger.info(f"Model path: {MODEL_DIR}")


def load_and_prepare(path):
    """Load and prepare data for Attention model"""
    logger.info("Loading data...")
    df = pd.read_csv(path)

    feature_cols = [
        'rate_of_interest', 'loan_amount', 'property_value',
        'income', 'Credit_Score', 'dtir1', 'term',
        'loan_limit', 'Gender', 'loan_type', 'loan_purpose',
        'Credit_Worthiness', 'open_credit', 'business_or_commercial',
        'Neg_ammortization', 'interest_only', 'lump_sum_payment',
        'construction_type', 'occupancy_type', 'Secured_by',
        'total_units', 'credit_type', 'co-applicant_credit_type',
        'age', 'submission_of_application', 'Region', 'Security_Type'
    ]
    target_col = 'Status'

    le = LabelEncoder()
    categorical = df[feature_cols].select_dtypes(include='object').columns
    for col in categorical:
        df[col] = le.fit_transform(df[col].astype(str))

    df = df[feature_cols + [target_col]].dropna()

    X = df[feature_cols].values
    y = df[target_col].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, MODEL_DIR / "attention_scaler.pkl")

    # Group features into 9 groups of 3 — gives attention meaningful structure
    # Each group = one "token" with 3 features
    X = X.reshape(X.shape[0], 9, 3)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train shape: {X_train.shape}")
    logger.info(f"Test shape: {X_test.shape}")
    logger.info(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test, scaler


def build_attention_model(input_shape):
    """Transformer-style Attention model"""
    inputs = Input(shape=input_shape)

    # Attention across feature groups
    attention_output = MultiHeadAttention(
        num_heads=3,
        key_dim=4
    )(inputs, inputs)

    x = LayerNormalization()(inputs + attention_output)

    # Second attention layer
    attention_output2 = MultiHeadAttention(
        num_heads=3,
        key_dim=4
    )(x, x)

    x = LayerNormalization()(x + attention_output2)

    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model


def train_attention_model(X_train, X_test, y_train, y_test):
    """Train the attention model"""
    logger.info("Building Attention model...")
    model = build_attention_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    logger.info(f"Class weights: {class_weight_dict}")

    early_stop = EarlyStopping(
        monitor='val_auc',
        patience=5,
        restore_best_weights=True,
        mode='max'
    )

    logger.info("Training Attention model...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=256,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        class_weight=class_weight_dict,
        verbose=1
    )

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_prob)
    logger.info(f"Attention Model ROC-AUC: {roc_auc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    model_path = MODEL_DIR / "attention_risk_model.keras"
    model.save(model_path)
    logger.info(f"Model saved to: {model_path}")

    return model, history


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_prepare(CLEANED_DATA_PATH)
    model, history = train_attention_model(X_train, X_test, y_train, y_test)