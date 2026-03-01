import pandas as pd
import joblib
import numpy as np
import os

def inspect():
    print("Script starting...", flush=True)
    # 1. Inspect CSV
    csv_path = r"c:\Users\ABY\Desktop\project\alzheimers\Data imputation KNN 2 neighbor\joined Labeled.csv"
    print(f"--- Inspecting CSV: {csv_path} ---")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            print(f"Columns: {df.columns.tolist()}")
            if 'Target' in df.columns:
                print("\nUnique values in 'Target' column:")
                try:
                    print(df['Target'].unique())
                    print("\nValue counts:")
                    print(df['Target'].value_counts())
                except Exception as e:
                    print(f"Error getting unique values: {e}")
            else:
                print("'Target' column not found.")
            
            print("\nFirst 5 rows:")
            print(df.head())
        except Exception as e:
            print(f"Error reading CSV: {e}")
    else:
        print("CSV file not found.")

    # 2. Inspect Model
    model_path = r"c:\Users\ABY\Desktop\project\alzheimers\results\global_model.joblib"
    print(f"\n--- Inspecting Model: {model_path} ---")
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print(f"Model type: {type(model)}")
            
            if hasattr(model, 'classes_'):
                print(f"Model classes_: {model.classes_}")
            elif hasattr(model, 'model') and hasattr(model.model, 'classes_'):
                 print(f"Wrapped Model classes_: {model.model.classes_}")
            else:
                print("Model does not have 'classes_' attribute.")

            if hasattr(model, 'n_classes_'):
                print(f"Model n_classes_: {model.n_classes_}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Model file not found.")

if __name__ == "__main__":
    inspect()
