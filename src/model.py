from sklearn.ensemble import RandomForestClassifier
import joblib

class RandomForestModel:
    def __init__(self, n_estimators=100, criterion='gini', max_depth=30,  random_state=42):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_params(self):
        """Returns the internal sklearn model."""
        return self.model

    def set_weights(self, trees):
        """
        For Federated Random Forest, 'setting weights' typically means 
        replacing the current forest's trees with the aggregated trees.
        """
        if trees:
             self.model.estimators_ = trees
             self.model.n_estimators = len(trees)
             

    def save(self, path):
        joblib.dump(self.model, path)
    
    def load(self, path):
        self.model = joblib.load(path)
