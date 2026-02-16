import os
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from typing import Optional

# Initialize FastAPI
app = FastAPI(title="Alzheimer's Disease Prediction")

# Templates
templates = Jinja2Templates(directory="templates")

# Configuration
DATA_PATH = "Datasets/Multimodal Dataset/phase1_processed.csv"
MODEL_PATH = "results/global_model.joblib"

# Global variables
model = None
scaler = None
feature_columns = None
explainer = None

class PatientData(BaseModel):
    LOGIMEM: float
    DIGIF: float
    DIGIFLEN: float
    DIGIB: float
    DIGIBLEN: float
    ANIMALS: float
    VEG: float
    TRAILA: float
    TRAILARR: float
    TRAILALI: float
    TRAILB: float
    TRAILBRR: float
    TRAILBLI: float
    WAIS: float
    MEMUNITS: float
    MEMTIME: float
    BOSTON: float
    IntraCranialVol: float
    lhCortexVol: float
    SubCortGrayVol: float
    SupraTentorialVol: float
    lhCorticalWhiteMatterVol: float
    mmse: float
    ageAtEntry: float
    commun: float
    homehobb: float
    judgment: float
    memory: float
    orient: float
    perscare: float
    apoe: float
    sumbox: float
    height: float
    weight: float

def load_resources():
    global model, scaler, feature_columns, explainer
    
    print("Loading resources...")
    
    if not os.path.exists(DATA_PATH):
        data_path = r"c:\Users\ABY\Desktop\project\alzheimers\Datasets\Multimodal Dataset\phase1_processed.csv"
    else:
        data_path = DATA_PATH
        
    print(f"Reading data from {data_path}")
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['Target'])
    feature_columns = X.columns.tolist()
    
    # Fit Scaler
    print("Fitting scaler...")
    scaler = StandardScaler()
    scaler.fit(X)
    
    print(f"Loading model from {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model_path = r"c:\Users\ABY\Desktop\project\alzheimers\results\global_model.joblib"
        model = joblib.load(model_path)
        
    try:
        import shap
        print("Initializing SHAP explainer...")
        if hasattr(model, 'model'):
             explainer = shap.TreeExplainer(model.model)
        else:
             explainer = shap.TreeExplainer(model)
    except Exception as e:
        print(f"Error initializing SHAP: {e}")

    print("Resources loaded successfully.")

@app.on_event("startup")
async def startup_event():
    load_resources()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(data: PatientData):
    global model, scaler, feature_columns, explainer
    
    if model is None or scaler is None:
        return {"error": "Model or Scaler not loaded"}
    
    input_dict = data.dict()
    df_input = pd.DataFrame([input_dict])
    
    required_cols = feature_columns
    
    for col in required_cols:
        if col not in df_input.columns:
            df_input[col] = 0.0
            
    df_input = df_input[required_cols]
    
    X_scaled = scaler.transform(df_input)
    
    if hasattr(model, 'predict'):
        prediction = int(model.predict(X_scaled)[0])
        probability = model.predict_proba(X_scaled)[0].tolist()
    else:
        prediction = int(model.predict(X_scaled)[0])
        probability = model.predict_proba(X_scaled)[0].tolist()
    
    class_map = {
        0: "Cognitively Normal (CN)",
        1: "Very Mild Dementia",
        2: "Mild Dementia",
        3: "Moderate Dementia",
        4: "Severe Dementia"
    }
    class_label = class_map.get(prediction, f"Class {prediction}")

    shap_plot_base64 = None
    if explainer:
        try:
            import shap
            import matplotlib.pyplot as plt
            import io
            import base64
            
            shap_values = explainer.shap_values(X_scaled)
            
            print(f"DEBUG: shap_values type: {type(shap_values)}")
            if isinstance(shap_values, list):
                print(f"DEBUG: shap_values is list of length {len(shap_values)}")
                print(f"DEBUG: shap_values[0] shape: {np.array(shap_values[0]).shape}")
            else:
                print(f"DEBUG: shap_values shape: {shap_values.shape}")
            
            if hasattr(explainer.expected_value, '__len__') and len(explainer.expected_value) > 1:
                expected_value = explainer.expected_value[prediction]
            else:
                expected_value = explainer.expected_value
                
            expected_value = float(expected_value)
            print(f"DEBUG: Final expected_value: {expected_value}")

            
            if isinstance(shap_values, list):
                
                shap_values_target = shap_values[prediction][0]
            elif len(shap_values.shape) == 3:
                
                shap_values_target = shap_values[0, :, prediction]
            else:
                shap_values_target = shap_values[0]
                
            print(f"DEBUG: Final shap_values_target shape: {shap_values_target.shape}")

            # Force Plot
            plt.figure(figsize=(20, 3))
            shap.force_plot(
                expected_value, 
                shap_values_target, 
                df_input.iloc[0], 
                feature_names=feature_columns,
                matplotlib=True,
                show=False
            )
            
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight', transparent=True)
            buf.seek(0)
            shap_plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            
            feature_importance = list(zip(feature_columns, shap_values_target))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_features = []
            for name, val in feature_importance[:5]:
                
                clean_name = name.replace("missingindicator_", "Missing Indicator: ")
                
                if val > 0:
                    impact = "increases the risk"
                    emoji = "⚠️"
                else:
                    impact = "decreases the risk"
                    emoji = "✅"
                    
                top_features.append(f"{emoji} {clean_name} {impact} (Impact: {val:.4f})")
            
        except Exception as e:
            print(f"SHAP Error: {e}")
            import traceback
            traceback.print_exc()
            top_features = [f"Could not generate explanations: {str(e)}"]
    
    return {
        "prediction": prediction,
        "probability": probability,
        "class_label": class_label,
        "shap_plot": shap_plot_base64,
        "explanations": top_features if explainer else []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
