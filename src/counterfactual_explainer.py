import dice_ml
from dice_ml import Dice
import pandas as pd
import numpy as np

class CounterfactualExplainer:
    def __init__(self, model, dataframe, target_column='Target'):
        """
        Initializes the Counterfactual Explainer.
        
        Args:
            model: The trained sklearn model (Random Forest).
            dataframe: The dataset used to define feature ranges and types.
            target_column: The name of the target column.
        """
        self.model = model
        self.df = dataframe
        self.target_column = target_column
        
        # Identify continuous and categorical features
        # For simplicity, we assume numeric features are continuous if they have many unique values
        self.continuous_features = []
        for col in self.df.columns:
            if col != self.target_column:
                if self.df[col].nunique() > 10:
                    self.continuous_features.append(col)
        
        # Setup DiCE data and model objects
        self.d = dice_ml.Data(
            dataframe=self.df,
            continuous_features=self.continuous_features,
            outcome_name=self.target_column
        )
        
        # DiCE model wrapper
        self.m = dice_ml.Model(model=self.model, backend="sklearn")
        
        # Initialize DiCE explainer
        self.exp = Dice(self.d, self.m, method="random")

    def generate_counterfactuals(self, query_instance, total_CFs=4, desired_class="opposite"):
        """
        Generates counterfactual explanations for a given input.
        
        Args:
            query_instance: A dataframe row (or dict) representing the patient data.
            total_CFs: Number of counterfactuals to generate.
            desired_class: The target class index or "opposite".
            
        Returns:
            A DiCE counterfactual object.
        """
        if isinstance(query_instance, dict):
            query_instance = pd.DataFrame([query_instance])
        
        # Find desired class if "opposite"
        if desired_class == "opposite":
            current_pred = self.model.predict(query_instance)[0]
            # Assumes 0, 1, 2 classes. If 1/2, go to 0. If 0, go to 1.
            desired_class = 0 if current_pred > 0 else 1
            
        dice_exp = self.exp.generate_counterfactuals(
            query_instance, 
            total_CFs=total_CFs, 
            desired_class=desired_class
        )
        return dice_exp

    def get_cf_dataframe(self, dice_exp):
        """Helper to extract the CFs as a dataframe."""
        return dice_exp.cf_examples_list[0].final_cfs_df
