import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)
    