import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Path configurations
DATA_PATH = r"c:\Users\ABY\Desktop\project\alzheimers\Datasets\Multimodal Dataset\phase1_processed.csv"
MODEL_PATH = r"c:\Users\ABY\Desktop\project\alzheimers\results\global_model.joblib"

def debug_prediction():
    print("--- Debugging Prediction ---")
    
    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    # Row 2 (Index 0 in pandas usually if header is row 1, but let's check content)
    # The user input was: LOGIMEM=9.0, DIGIF=8.0...
    # In the file view earlier:
    # 1: LOGIMEM...
    # 2: 9.0,8.0... (This is the first data row)
    
    target_row = df.iloc[0] # The first data row
    print("Selected Row Target:", target_row['Target'])
    
    # 2. Scaler Logic (Same as DataLoader)
    X = df.drop(columns=['Target'])
    scaler = StandardScaler()
    scaler.fit(X)
    
    # 3. Transform the specific row
    # We need to reshape to (1, -1) for a single sample
    X_sample_raw = X.iloc[[0]] 
    X_sample_scaled = scaler.transform(X_sample_raw)
    
    # 4. Load Model
    model = joblib.load(MODEL_PATH)
    
    # 5. Predict
    prediction = model.predict(X_sample_scaled)[0]
    probabilities = model.predict_proba(X_sample_scaled)[0]
    
    print(f"Model Prediction: {prediction}")
    print(f"Model Probabilities: {probabilities}")
    
    if prediction != target_row['Target']:
        print("MISMATCH: Model predicts differently than label. This confirms it's a model issue (or feature of the probabilistic model), not necessarily an API bug.")
    else:
        print("MATCH: Model predicts correctly.")

if __name__ == "__main__":
    debug_prediction()
