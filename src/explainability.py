import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class Explainability:
    def __init__(self, model, X_train):
        """
        Initializes the Explainability module.
        
        Args:
            model: The trained model (sklearn wrapper or compatible).
            X_train: Background data for SHAP (subset of training data).
        """
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None

    def calculate_shap_values(self):
        """Calculates SHAP values using TreeExplainer."""
        print("Calculating SHAP values...")
        try:
            if hasattr(self.model, 'model'):
                estimator = self.model.model
            else:
                estimator = self.model

            self.explainer = shap.TreeExplainer(estimator)
            self.shap_values = self.explainer.shap_values(self.X_train)
            print("SHAP values calculated successfully.")
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")

    def plot_summary(self, save_path="shap_summary_plot.png"):
        """Generates and saves a SHAP summary plot."""
        if self.shap_values is None:
            self.calculate_shap_values()
            
        plt.figure()
        vals = self.shap_values
        if isinstance(vals, list):
            vals = vals[1]
            
        shap.summary_plot(vals, self.X_train, show=False)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Summary plot saved to {save_path}")

    def plot_violin(self, feature_names=None, save_path="shap_violin_plot.png"):
        """Generates and saves a SHAP violin plot."""
        if self.shap_values is None:
            self.calculate_shap_values()

        plt.figure()
        vals = self.shap_values
        if isinstance(vals, list):
            vals = vals[1]

        shap.summary_plot(vals, self.X_train, plot_type="violin", show=False)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Violin plot saved to {save_path}")
