cat > moghedien/core/models.py << 'EOL'
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

class AttackPathModel:
    """
    Base class for attack path prediction models.
    """
    def __init__(self):
        pass
    
    def predict(self, path_features: Dict[str, Any]) -> float:
        """
        Predict the score for an attack path.
        
        Args:
            path_features: Dictionary of path features
            
        Returns:
            Score for the path (higher is better)
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def train(self, paths: List[Dict[str, Any]], scores: List[float]) -> None:
        """
        Train the model on a set of paths.
        
        Args:
            paths: List of path feature dictionaries
            scores: List of path scores
        """
        raise NotImplementedError("Subclasses must implement train method")


class RandomForestPathModel(AttackPathModel):
    """
    A random forest model for predicting attack path scores.
    """
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10)
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.feature_names = []
EOL
