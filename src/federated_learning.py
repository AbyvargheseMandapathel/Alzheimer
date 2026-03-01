import numpy as np
import copy
from typing import List, Any
from src.model import RandomForestModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class FederatedClient:
    def __init__(self, client_id: int, X_train, y_train):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.model = RandomForestModel() # Local model initialization

    def train(self, round_seed=None):
        """
        Trains the local model and returns its trees (parameters).
        Args:
            round_seed: Optional seed to ensure variation across rounds.
        """
        print(f"Client {self.client_id}: Training...")
        if round_seed is not None:
            self.model.model.random_state = (self.model.random_state + round_seed) % 10000
            
        self.model.fit(self.X_train, self.y_train)
        return self.model.get_params().estimators_

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return acc

class FederatedServer:
    def __init__(self, global_model: RandomForestModel):
        self.global_model = global_model
        self.global_trees = []

    def aggregate(self, client_trees_list: List[List[Any]]):
        """
        Aggregates client models. 
        For Random Forest, a common FL approach is to aggregate all trees 
        into a global ensemble (Federated Forest).
        """
        print("Server: Aggregating models...")
        self.global_trees = []
        for trees in client_trees_list:
            self.global_trees.extend(trees)
        
        # Update global model with aggregated trees
        self.global_model.set_weights(self.global_trees)
        
        # Determine classes from one of the estimators if not already set (hack for sklearn check)
        if self.global_trees and not hasattr(self.global_model.model, 'classes_'):
             self.global_model.model.classes_ = self.global_trees[0].classes_
             self.global_model.model.n_classes_ = self.global_trees[0].n_classes_
             self.global_model.model.n_outputs_ = self.global_trees[0].n_outputs_

    def evaluate(self, X_test, y_test):
        print("Server: Evaluating global model...")
        y_pred = self.global_model.predict(X_test)
        y_prob = self.global_model.predict_proba(X_test)
        
        if len(np.unique(y_test)) == 2:
             auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
             try:
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
             except:
                auc = 0.0 

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'auc': auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            # 'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        return metrics
