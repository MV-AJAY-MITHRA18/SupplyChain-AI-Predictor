import pandas as pd
import numpy as np
import joblib
import shap
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report

# --- Configuration ---
DATA_PATH = 'data/simulated_shipments.csv'
XGB_MODEL_PATH = 'models/xgboost_pipeline.joblib'
LSTM_MODEL_PATH = 'models/lstm_model.h5'
PREPROCESSOR_PATH = 'models/preprocessor.joblib'

# FEATURES used for the model
MODEL_FEATURES = [
    'Origin_WH', 'Destination_City', 'Product_Category', 'Shipping_Mode', 
    'Warehouse_Congestion', 'Strike_Flag', 'Weather_Severity', 
    'Customs_Delay_Flag', 'Scheduled_Transit_Days', 'Departure_DayOfWeek'
]
TARGET = 'Arrival_Delay_Status' 

os.makedirs('models', exist_ok=True)
tf.random.set_seed(42)

# 1. Data Loading and Preprocessing
print("1. Loading and Preprocessing Data...")
df = pd.read_csv(DATA_PATH)

# Separate ID and Target from Features
ID_COL = 'OrderID'
X = df[[ID_COL] + MODEL_FEATURES] # Keep OrderID for splitting
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Extract OrderID columns for later merging
X_test_ids = X_test[[ID_COL]].copy().reset_index(drop=True)
X_train_features = X_train[MODEL_FEATURES]
X_test_features = X_test[MODEL_FEATURES]


# Define preprocessing steps
numerical_features = ['Warehouse_Congestion', 'Weather_Severity', 'Scheduled_Transit_Days']
categorical_features = ['Origin_WH', 'Destination_City', 'Product_Category', 'Shipping_Mode', 'Departure_DayOfWeek']
binary_features = ['Strike_Flag', 'Customs_Delay_Flag']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# Fit preprocessor on training features and transform
X_train_processed = preprocessor.fit_transform(X_train_features)
X_test_processed = preprocessor.transform(X_test_features)
joblib.dump(preprocessor, PREPROCESSOR_PATH)
print(f"✅ Preprocessor saved to {PREPROCESSOR_PATH}")


# 2. Model 1: XGBoost (Prediction and SHAP)
print("\n2. Training XGBoost Classifier...")
xgb_model = XGBClassifier(
    eval_metric='logloss', 
    n_estimators=400, 
    max_depth=5, 
    learning_rate=0.05, 
    random_state=42, 
    scale_pos_weight=y.value_counts()[0]/y.value_counts()[1]
)
xgb_model.fit(X_train_processed, y_train)
y_pred_xgb = xgb_model.predict(X_test_processed)
print("--- XGBoost Performance ---")
print(classification_report(y_test, y_pred_xgb))
joblib.dump(xgb_model, XGB_MODEL_PATH)
print(f"✅ XGBoost model saved to {XGB_MODEL_PATH}")


# 3. Model 2: Deep Learning (Keras/TensorFlow)
print("\n3. Training Deep Learning (Keras/TensorFlow) Model...")
def create_dl_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall()])
    return model

dl_model = KerasClassifier(
    model=create_dl_model,
    model__input_dim=X_train_processed.shape[1],
    epochs=50, batch_size=128, verbose=0, random_state=42
)
dl_model.fit(X_train_processed, y_train.values)
dl_model.model_.save(LSTM_MODEL_PATH)


# 4. Explainable AI (SHAP) - Advanced Feature
print("\n4. Explainable AI (SHAP) Analysis on XGBoost...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_processed)

feature_names = (
    numerical_features + 
    list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)) +
    binary_features
)

# Generate output dataframe for the next stage (Optimization)
X_test_output = X_test_ids.copy()
X_test_output['Predicted_Delay_Probability'] = xgb_model.predict_proba(X_test_processed)[:, 1]

# Extract Top 3 SHAP drivers for each order
top_3_features_indices = np.argsort(np.abs(shap_values), axis=1)[:, ::-1][:, :3]
top_3_features = [", ".join([feature_names[i] for i in row]) for row in top_3_features_indices]

X_test_output['Top_Delay_Drivers'] = top_3_features

# --- FINAL MERGE CORRECTION ---
# Merge Cost_Impact from the original full dataframe (df) using the OrderID
# Drop the index column from df before merging to ensure a clean merge on OrderID
df_cost_data = df[[ID_COL, 'Cost_Impact_USD']].drop_duplicates().reset_index(drop=True)

final_output_df = X_test_output.merge(df_cost_data, on=ID_COL, how='left')


final_output_df.to_csv('data/prediction_output.csv', index=False)
print("✅ Prediction data (with probability and SHAP drivers) saved for optimization.")