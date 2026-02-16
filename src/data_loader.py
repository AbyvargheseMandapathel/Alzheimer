import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict

class DataLoader:
    def __init__(self, file_path: str, target_column: str = 'Target', n_clients: int = 10, random_state: int = 42):
        """
        Initialize the DataLoader.

        Args:
            file_path: Path to the CSV file.
            target_column: Name of the target label column.
            n_clients: Number of clients for Federated Learning simulation.
            random_state: Seed for reproducibility.
        """
        self.file_path = file_path
        self.target_column = target_column
        self.n_clients = n_clients
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Loads the dataset from the CSV file."""
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            
            if self.data.isnull().sum().sum() > 0:
                print("Warning: Missing values found. Dropping rows with missing values...")
                self.data.dropna(inplace=True)
                
            self.X = self.data.drop(columns=[self.target_column])
            self.y = self.data[self.target_column]
            
            self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=self.X.columns)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at {self.file_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def split_data(self, test_size: float = 0.2):
        """Splits data into global training and testing sets."""
        if self.X is None or self.y is None:
            self.load_data()
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state, stratify=self.y
        )
        print(f"Data split: Train shape {self.X_train.shape}, Test shape {self.X_test.shape}")

    def get_client_data(self) -> List[Tuple[pd.DataFrame, pd.Series]]:
        """
        Simulates data partitioning for Federated Learning clients.
        Currently implements IID splitting (random shuffling).
        
        Returns:
            A list of tuples, where each tuple contains (X_client, y_client) for a client.
        """
        if self.X_train is None:
            self.split_data()

        train_data = pd.concat([self.X_train, self.y_train], axis=1)
        
        train_data = train_data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        chunk_size = len(train_data) // self.n_clients
        client_datasets = []
        
        for i in range(self.n_clients):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < self.n_clients - 1 else len(train_data)
            
            chunk = train_data.iloc[start:end]
            
            X_c = chunk.drop(columns=[self.target_column])
            y_c = chunk[self.target_column]
            client_datasets.append((X_c, y_c))
            
        print(f"Data partitioned into {self.n_clients} clients.")
        return client_datasets

if __name__ == "__main__":
    path = "c:\\Users\\ABY\\Desktop\\project\\alzheimers\\Datasets\\Multimodal Dataset\\Multimodal After data imputation.csv"
    loader = DataLoader(path)
    loader.load_data()
    loader.split_data()
    clients = loader.get_client_data()
    print(f"Client 1 Data Shape: {clients[0][0].shape}")
