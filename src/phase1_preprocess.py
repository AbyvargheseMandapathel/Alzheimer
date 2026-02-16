import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def run_phase1_preprocessing():
    print("Starting Phase 1: Data Preprocessing & Imputation...")
    
    # 1. Configuration
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_FILE = os.path.join(BASE_DIR, "Data imputation KNN 2 neighbor", "joined Labeled.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "Datasets", "Multimodal Dataset")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phase1_processed.csv")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        if os.path.exists("Data imputation KNN 2 neighbor/joined Labeled.csv"):
             INPUT_FILE = "Data imputation KNN 2 neighbor/joined Labeled.csv"
        else:
             return

    # 2. Load Data
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, na_values=' ?') 
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    print(f"Initial shape: {df.shape}")
    
    # 3. Preprocessing
    if 'dx1' in df.columns:
        print("Dropping 'dx1' column...")
        X = df.drop(columns=['dx1', 'Target'])
        y = df['Target']
    else:
        print("Warning: 'dx1' column not found.")
        X = df.drop(columns=['Target'])
        y = df['Target']
        
    
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Identified {len(numerical_cols)} numerical columns for imputation: {numerical_cols}")
    
    # 4. KNN Imputation
    print("Applying KNN Imputation (n_neighbors=2)...")
    knn_imputer = KNNImputer(n_neighbors=2, add_indicator=True)
    
    # Fit and transform on numerical columns
    X_filled_num = knn_imputer.fit_transform(X[numerical_cols])
    
    
    try:
        new_cols = knn_imputer.get_feature_names_out(numerical_cols)
    except AttributeError:
        # Fallback for older sklearn
        new_cols = numerical_cols # Approximate, might miss indicators if implied
        if X_filled_num.shape[1] > len(numerical_cols):
             # Just name them generically if we can't easily get names
             extra = X_filled_num.shape[1] - len(numerical_cols)
             new_cols = list(numerical_cols) + [f"missing_ind_{i}" for i in range(extra)]

    X_processed = pd.DataFrame(X_filled_num, columns=new_cols)
    

    
    print(f"Shape after Imputation: {X_processed.shape}")

    # 4b. Feature Selection (Pearson Correlation)
    print("Applying Pearson Correlation Feature Selection (threshold > 0.95)...")
    corr_matrix = X_processed.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    if to_drop:
        print(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
        X_processed = X_processed.drop(columns=to_drop)
    else:
        print("No features dropped (all correlations <= 0.95).")
    
    print(f"Shape after Feature Selection: {X_processed.shape}")

    # 5. SMOTE Balancing
    print("Applying SMOTE balancing...")
    smote = SMOTE(random_state=42)
    try:
        X_resampled, y_resampled = smote.fit_resample(X_processed, y)
        print(f"Shape after SMOTE: {X_resampled.shape}")
        
        # Combine back into a single dataframe
        final_df = pd.concat([pd.DataFrame(X_resampled, columns=X_processed.columns), 
                              pd.Series(y_resampled, name='Target')], axis=1)
        
        # 6. Save Data
        print(f"Saving processed data to {OUTPUT_FILE}...")
        final_df.to_csv(OUTPUT_FILE, index=False)
        print("Phase 1 completed successfully.")
        
    except Exception as e:
        print(f"Error during SMOTE: {e}")

if __name__ == "__main__":
    run_phase1_preprocessing()
