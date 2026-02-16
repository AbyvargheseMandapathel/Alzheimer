import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = "results/global_model.joblib"
DATA_PATH = "Datasets/Multimodal Dataset/phase1_processed.csv"

def analyze_features():
    print("Loading model and data...")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['Target'])
    feature_names = X.columns.tolist()

    # Get Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
         importances = model.model.feature_importances_
    else:
        print("Model does not expose feature_importances_")
        return

    # Create DataFrame
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

    print("\nTop 20 Features:")
    print(feature_imp_df.head(20))
    
    # Calculate cumulative importance
    feature_imp_df['Cumulative'] = feature_imp_df['Importance'].cumsum()
    print("\nCumulative Importance:")
    print(feature_imp_df.head(20)[['Feature', 'Importance', 'Cumulative']])
    
    # Identify 95% threshold
    threshold_idx = feature_imp_df[feature_imp_df['Cumulative'] > 0.95].index[0]
    num_features_95 = feature_imp_df.index.get_loc(threshold_idx) + 1
    print(f"\nNumber of features needed for 95% importance: {num_features_95}")

if __name__ == "__main__":
    analyze_features()
