import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import DataLoader
from src.model import RandomForestModel
from src.federated_learning import FederatedServer, FederatedClient
from src.explainability import Explainability
import warnings

# Suppress Warnings
warnings.filterwarnings("ignore")

def main():
    print("Starting Federated Alzheimer's Disease Prediction...")
    
    # 1. Configuration
    DATA_PATH = "Datasets/Multimodal Dataset/phase1_processed.csv"
    N_CLIENTS = 10
    ROUNDS = 50 
    RESULTS_DIR = "results"
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # 2. Data Loading
    print("\n--- Data Loading ---")
    # Resolve absolute path if needed, or rely on relative if running from root
    if not os.path.exists(DATA_PATH):
        # Fallback for checking absolute path from knowledge context
        DATA_PATH = r"c:\Users\ABY\Desktop\project\alzheimers\Datasets\Multimodal Dataset\Multimodal After data imputation.csv"
        
    loader = DataLoader(DATA_PATH, n_clients=N_CLIENTS)
    loader.load_data()
    loader.split_data() # Creates global train/test split
    client_datasets = loader.get_client_data()
    
    # 3. Initialization
    print("\n--- Initialization ---")
    # Create Global Server Model
    global_model = RandomForestModel(n_estimators=100) # Initial empty model logic handled in aggregation
    server = FederatedServer(global_model)
    
    # Create Clients
    clients = []
    for i in range(N_CLIENTS):
        X_c, y_c = client_datasets[i]
        client = FederatedClient(client_id=i, X_train=X_c, y_train=y_c)
        clients.append(client)
        
    print(f"Initialized Server and {len(clients)} Clients.")

    # 4. Federated Training Loop
    print("\n--- Federated Training Loop ---")
    metrics_history = []
    
    for round_num in range(1, ROUNDS + 1):
        print(f"\nRound {round_num}/{ROUNDS}")
        
        # A. Client Training
        client_trees = []
        for client in clients:
            # Simulated FL: Client trains local model with a varying seed
            trees = client.train(round_seed=round_num)
            # client.evaluate(loader.X_test, loader.y_test) # Optional local eval
            client_trees.append(trees)
            
        # B. Server Aggregation
        server.aggregate(client_trees)
        
        # C. Global Evaluation
        metrics = server.evaluate(loader.X_test, loader.y_test)
        metrics['round'] = round_num
        metrics_history.append(metrics)
        
        print(f"Round {round_num} Metrics: Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
        
        # Save Confusion Matrix for the last round (or best round)
        if round_num == ROUNDS:
            cm = np.array(metrics['confusion_matrix'])
            print("Final Confusion Matrix:\n", cm)
            np.save(os.path.join(RESULTS_DIR, "final_confusion_matrix.npy"), cm)
            
            # Simple Plot using matplotlib (seaborn is better but keeping dependencies minimal if needed)
            import seaborn as sns
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Final Confusion Matrix')
            plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
            plt.close()

    # 5. Results and Visualization
    print("\n--- Saving Results ---")
    metrics_df = pd.DataFrame(metrics_history)
    metrics_csv_path = os.path.join(RESULTS_DIR, "training_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics saved to {metrics_csv_path}")
    
    # Plot Accuracy over Rounds
    plt.figure()
    plt.plot(metrics_df['round'], metrics_df['accuracy'], label='Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Federated Learning Accuracy over Rounds')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_plot.png"))
    plt.close()

    # 6. Save Global Model
    print("\n--- Saving Global Model ---")
    model_save_path = os.path.join(RESULTS_DIR, "global_model.joblib")
    server.global_model.save(model_save_path)
    print(f"Global model saved to {model_save_path}")

    # 7. Explainability
    print("\n--- Explainability ---")
    # We use the global model and a subset of test data for explanation
    explainer = Explainability(server.global_model, loader.X_test)
    explainer.calculate_shap_values()
    explainer.plot_summary(save_path=os.path.join(RESULTS_DIR, "shap_summary.png"))
    explainer.plot_violin(save_path=os.path.join(RESULTS_DIR, "shap_violin.png"))
    
    print("\nDone! Workflows completed.")

if __name__ == "__main__":
    main()
