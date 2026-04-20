import torch
import pandas as pd
import numpy as np
import joblib
import os
from src.counterfactual_explainer import CounterfactualExplainer
from explainability.counterfactual_image import ImageCounterfactual
from models.cnn_model import AlzheimerResNet

def test_tabular_cf():
    print("\n--- Testing Tabular Counterfactuals ---")
    model_path = "results/global_model.joblib"
    data_path = "Datasets/Multimodal Dataset/phase1_processed.csv"
    
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print("Model or data not found. Skipping tabular test.")
        return

    rf_model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    explainer = CounterfactualExplainer(rf_model, df)
    
    # Test with a sample from Class 1 or 2
    query = df[df['Target'] > 0].iloc[[0]].drop(columns=['Target'])
    print(f"Current Prediction: {rf_model.predict(query)[0]}")
    
    cf_result = explainer.generate_counterfactuals(query, total_CFs=1, desired_class=0)
    cf_df = explainer.get_cf_dataframe(cf_result)
    
    print("Counterfactual generated successfully.")
    print(f"Counterfactual Target: {cf_df['Target'].iloc[0]}")

def test_image_cf():
    print("\n--- Testing Image Counterfactuals ---")
    device = "cpu"
    model = AlzheimerResNet(num_classes=3).to(device)
    # Use dummy weights since we just want to test logic
    img_t = torch.randn(1, 3, 224, 224).to(device)
    
    cf_generator = ImageCounterfactual(model, device)
    cf_map = cf_generator.generate_counterfactual_map(img_t, target_class_idx=0)
    
    print(f"Counterfactual map shape: {cf_map.shape}")
    print(f"Counterfactual map max value: {cf_map.max()}")
    if cf_map.shape == (224, 224):
        print("Image Counterfactual logic verified.")

if __name__ == "__main__":
    test_tabular_cf()
    test_image_cf()
